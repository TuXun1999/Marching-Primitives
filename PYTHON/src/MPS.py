import cupy as cp
import numpy as np
import argparse
import skimage
import skimage.measure
from sklearn.neighbors import KDTree
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares


def MPS(sdf, grid, para):
    """
    Weixiao Liu 2023 Johns Hopkins University/National University of Singapore
    
    Modified: Xun Tu, 2024, a humble PhD student trying to produce at least some
    junk to graduate
    Xun's Jobs: Convert the original Matlab codes into Python
    Parameters:
    sdf (numpy.ndarray): A vector containing the flattened truncated signed distance field
    grid (dict): An object containing the gridding info of the TSDF
    **kwargs: Algorithm hyperparameters
    
    Returns:
    numpy.ndarray: The superquadric configuration
    """
    
    # Two major differences between Matlab & Python:
    # Order Difference:
    # python: C
    # matlabl: F
    # One typical example is the Bounding Box Convention:
    # python: min_x, min_y, min_z, max_x, max_y, max_z
    # matlab: min_y, min_x, min_z, length_x, length_y, length_z

    # Starting point Convention:
    # python: starts at 0
    # matlab: starts at 1
    
    # Initialization
    numDivision = 1  # Initialize the depth of subdivision
    x = []  # Declare the array storing the superquadric configuration
    dratio = 3/5  # 4/5
    connRatio = [1, dratio, dratio**2, dratio**3, dratio**4, 
                 dratio**5, dratio**6, dratio**7, dratio**8]
    connpointer = 0
    num_region = 1
    # TODO: Organize the codes if things are working well!!
    while numDivision < para.maxDivision:
        # terminate when the maximum layer of division reached 
        # 1-Connectivity Marching--------------------------------------------------
        if connpointer != 0 and num_region != 0:
            connpointer = 0
        # set the connection region 
        connTreshold = connRatio[connpointer] * min(sdf)

        if connTreshold > -grid["truncation"] * 3e-1:
            break
        # preliminary segementation via connectivity of regions
        # rearrange sdf to 3D array for region connection checking
        sdf3d_region = np.reshape(sdf, \
                    (grid["size"][0], grid["size"][1], grid["size"][2]), \
                    order='F')
        # connection checking and preliminary subdivision
        region_filter = sdf3d_region <= connTreshold
        region_filter = region_filter.astype(int)
        region_filter = np.array(region_filter)

        # Calculate the regions of interest
        # TODO: figure out the different ordering order & the -1 index
        roi_props = skimage.measure.regionprops(skimage.measure.label(region_filter.astype(int)))
        #                properties=("coords", "area", "centroid", "bbox"))
        # calculate the size of interest regions 
        roi_props = [r for r in roi_props if r["area"]>= para.minArea]
        num_region = len(roi_props)

        if para.verbose == 1:
            print(f"Number of regions: {num_region}")

        if num_region == 0:
            if connpointer < len(connRatio) - 1: # len(connRation) - 1 in Matlab codes...
                connpointer += 1
                continue
            else:
                break

        # 2-Probabilistic Primitive Marching---------------------------------------
        ## initialize superquadric storage
        x_temp = np.zeros((num_region, 11))
        del_idx = np.zeros(num_region)
        occ_idx_in = [[] for _ in range(num_region)]
        num_idx = np.zeros((num_region, 3))
        roi_regions = []
        for i in range(num_region):
            occ_idx = []

            ## Padding the bounding box to extend the region of interest
            # also update the bounding box info accordingly


            # Temporary variable to store the attributes of that roi
            # (Regionproperties in Python doesn't support item assignment)
            roi = {}
            idx = np.ceil(roi_props[i]["bbox"]) 
            idx[3:6] = np.minimum(idx[3:6] + para.paddingSize, \
                                  [grid["size"][0] - 1, grid["size"][1] - 1, grid["size"][2]- 1] )
            idx[0:3] = np.maximum(idx[0:3] - para.paddingSize, 0)
            idx_x, idx_y, idx_z = np.meshgrid(np.arange(idx[0], idx[3]+1), \
                                              np.arange(idx[1], idx[4]+1), \
                                                np.arange(idx[2], idx[5]+1))
            roi["bbox"]= idx

            ## Find the voxels activated, that is, within the bounding box
            # Permute the order
            idx_x = np.transpose(idx_x, (1, 0, 2))
            idx_y = np.transpose(idx_y, (1, 0, 2))
            idx_z = np.transpose(idx_z, (1, 0, 2))

            # Construct the points in voxelGrid (obey the convention in original Matlab codes)
            indices = np.stack([idx_x, idx_y, idx_z], axis=3)
            indices = indices.reshape(-1, 3, order='F').T.astype(int)

            # Indices & Point coordinates that are activated in each subdivision
            roi["idx"] = idx3d_flatten(indices, grid)
            roi["bounding_points"] = idx2Coordinate(np.array([
                        [idx[0], idx[0], idx[3], idx[3], idx[0], idx[0], idx[3], idx[3]],
                        [idx[1], idx[1], idx[1], idx[1], idx[4], idx[4], idx[4], idx[4]],
                        [idx[2], idx[5], idx[2], idx[5], idx[2], idx[5], idx[2], idx[5]]]),
                                                grid)
            
            ## Centralize the initial position inside the volumetric region
            # I actually feel that this step may be redundant in Python...                       
            roi["centroid"] = np.maximum(np.floor(roi_props[i]["centroid"]), 0) 
            centroid_column = idx3d_flatten(roi["centroid"], grid).astype(int)

            # Determine the coordinate (instead of just indices) of the centroid
            point_idx_in_region = np.array(roi_props[i].coords).T
            point_column_in_region = idx3d_flatten(point_idx_in_region, grid).astype(int) # 1D vector
            roi["point_column_in_region"] = point_column_in_region
            # Check if the center of the region is already assigned a sdf value
            if centroid_column in point_column_in_region:
                roi["centroid"]= grid["points"][:, centroid_column]
            else:
                # point_column_in_region = np.sort(point_column_in_region)
                # search for nearest inside point within the roi
                query_point = grid["points"][:, centroid_column]
                search_point = grid["points"][:, point_column_in_region]

                # TODO: test one thing in the future!!
                # Probably due to different data format order, 
                # the two methods will generate different closest point indices to the same query point
                # since there are indeed multiple solutions to the closest point problem
                kdt = KDTree(search_point.T)
                k = kdt.query(query_point.reshape(1, -1), k=1, \
                                 return_distance = False, breadth_first=False)
                roi["centroid"] = search_point[:, k[0]].flatten()

                # TODO: debug purpose
                # roi["centroid"] = grid["points"][:, 685634]
                # k = np.argmin(np.linalg.norm(search_point - query_point.reshape(-1,1), axis=0))
                # roi["centroid"] = search_point[:, k]
        
            ## Reconstruct sq and check the precision
            valid = np.zeros(6)
            while not all(valid):
                # initialize superquadric scale
                scale_init = para.scaleInitRatio * (roi["bbox"][3:6] - \
                    roi["bbox"][0:3]) * grid["interval"]
                
                x_init = np.array(\
                    [1, 1, scale_init[0], scale_init[1], scale_init[2], \
                    0, 0, 0, roi["centroid"][0], roi["centroid"][1], roi["centroid"][2]])
                
                # TODO: Complete this part!!
                # TODO: may want to test the correctness with user-specified input
            #      % for each subdivision find the optimal superquadric representation
                
                x_temp_i, occ_idx, valid, num_idx_i = fitSuperquadricTSDF(\
                    sdf[roi["idx"]], \
                    x_init, \
                    grid["truncation"], \
                    grid["points"][:, roi["idx"]], \
                    roi["idx"], \
                    roi["bounding_points"], \
                    para)
                x_temp[i, :] = x_temp_i
                num_idx[i, :] = num_idx_i
                if not all(valid):
                    # TODO: figure out what this part does more clearly
                    extense = np.logical_not(valid)


                    # If the boundary is reached, break
                    if any([any(idx[:3][extense[0:3]] == 0),\
                        idx[3] == grid["size"][0] - 1 and extense[3], \
                        idx[4] == grid["size"][1] - 1 and extense[4], \
                        idx[5] == grid["size"][2] - 1 and extense[5]]):
                        break

                    # TODO: figure out what this part does
                    # Extend to a new region?
                    idx_extend = extense * para.paddingSize
                    idx[3:6] = np.minimum(idx[3:6] + idx_extend[3:6], \
                        [grid["size"][0] - 1, grid["size"][1] - 1, grid["size"][2] - 1])
                    idx[0:3] = np.maximum(idx[0:3]- idx_extend[0:3], 0)
                    idx_x, idx_y, idx_z = np.meshgrid(\
                        np.arange(idx[0],idx[3]+1),\
                        np.arange(idx[1],idx[4]+1),\
                        np.arange(idx[2],idx[5]+1)
                    )
                    
                    ## Update roi within this while loop
                    roi["bounding_box"]= idx
                    ## Find the voxels activated, that is, within the bounding box
                    # Permute the order
                    idx_x = np.transpose(idx_x, (1, 0, 2))
                    idx_y = np.transpose(idx_y, (1, 0, 2))
                    idx_z = np.transpose(idx_z, (1, 0, 2))

                    # Construct the points in voxelGrid (obey the convention in original Matlab codes)
                    indices = np.stack([idx_x, idx_y, idx_z], axis=3)
                    indices = indices.reshape(-1, 3, order='F').T.astype(int)

                    # Indices & Point coordinates that are activated in each subdivision
                    roi["idx"] = idx3d_flatten(indices, grid)
                    roi["bounding_points"] = idx2Coordinate(np.array([
                                [idx[0], idx[0], idx[3], idx[3], idx[0], idx[0], idx[3], idx[3]],
                                [idx[1], idx[1], idx[1], idx[1], idx[4], idx[4], idx[4], idx[4]],
                                [idx[2], idx[5], idx[2], idx[5], idx[2], idx[5], idx[2], idx[5]]]),
                                                        grid)         
                       
            occ_idx_in[i] = occ_idx[sdf[occ_idx] <= 0]
            roi_regions.append(roi)

        ## Evaluate the fitting quality
        # TODO: figure out what these parameters mean
        for i in range(num_region):
            # evaluate fitting quality            
            if (num_idx[i, 1] / (num_idx[i, 0] + num_idx[i, 1])) > 0.3 \
                    or num_idx[i, 0] < para.minArea or num_idx[i, 2] <= 1: #1:
                del_idx[i] = 1
                sdf[roi_regions[i]["point_column_in_region"]] = np.nan
                if para.verbose == 1:
                    print(['region ' + str(i) + '/'+ str(num_region) + \
                        ' outPrecentage: ' + \
                        str(num_idx[i, 1] / (num_idx[i, 0] + num_idx[i, 1])) + \
                        ' inNumber: ' + \
                        str(num_idx[i, 2]) + \
                        ' ...REJECTED'])
            else:
                sdf[occ_idx_in[i]] = np.nan
                if para.verbose == 1:
                    print(['region ' + str(i) + '/'+ str(num_region) + \
                        ' outPrecentage: ' + \
                        str(num_idx[i, 1] / (num_idx[i, 0] + num_idx[i, 1])) + \
                        ' inNumber: ' + \
                        str(num_idx[i, 2]) + \
                        ' ...ACCEPTED'])

        x_temp = x_temp[del_idx == 0, :]
        x.append(x_temp)
        numDivision = numDivision + 1
    return np.concatenate(x, axis=0)

def parseInputArgs(grid, argv):
    '''
    The function to parses the inputs
    '''
    # set input parser
    defaults = {
        'verbose': True,
        'paddingSize': np.ceil(12 * grid["truncation"] / grid["interval"]),
        'minArea': np.ceil(grid["size"][0] / 20),
        'maxDivision': 50,
        'scaleInitRatio': 0.1,
        'nanRange': 0.5 * grid["interval"],
        'w': 0.99,
        'tolerance': 1e-6,
        'relative_tolerance': 1e-4,
        'switch_tolerance': 1e-1,
        'maxSwitch': np.uint8(2),
        'iter_min': np.uint8(2),
        'maxOptiIter': 4, # One default parameter is changed here
        'maxIter': 15,
        'activeMultiplier': 3
    }

    parser = argparse.ArgumentParser(
        description="Marching-Primitive Algorithm to split the mesh into several parts"
    )
    parser.add_argument('--verbose', type=bool, default=defaults['verbose'])
    parser.add_argument('--paddingSize', type=float, default=defaults['paddingSize'])
    parser.add_argument('--minArea', type=float, default=defaults['minArea'])
    parser.add_argument('--maxDivision', type=float, default=defaults['maxDivision'])
    parser.add_argument('--scaleInitRatio', type=float, default=defaults['scaleInitRatio'])
    parser.add_argument('--nanRange', type=float, default=defaults['nanRange'])
    parser.add_argument('--w', type=float, default=defaults['w'])
    parser.add_argument('--tolerance', type=float, default=defaults['tolerance'])
    parser.add_argument('--relative_tolerance', type=float, default=defaults['relative_tolerance'])
    parser.add_argument('--switch_tolerance', type=float, default=defaults['switch_tolerance'])
    parser.add_argument('--maxSwitch', type=int, default=defaults['maxSwitch'])
    parser.add_argument('--iter_min', type=int, default=defaults['iter_min'])
    parser.add_argument('--maxOptiIter', type=int, default=defaults['maxOptiIter'])
    parser.add_argument('--maxIter', type=int, default=defaults['maxIter'])
    parser.add_argument('--activeMultiplier', type=float, default=defaults['activeMultiplier'])

    args = parser.parse_args(argv)
    return args


def idx2Coordinate(idx, grid):
    '''
    The function to convert the indices of the bounding points into 
    3D coordinates
    '''
    idx_floor = np.floor(idx).astype(int)
    idx_floor[idx_floor == -1] = 0

    x = grid["x"][idx_floor[0, :]] + (idx[0, :] - idx_floor[0, :]) * grid["interval"]
    y = grid["y"][idx_floor[1, :]] + (idx[1, :] - idx_floor[1, :]) * grid["interval"]
    z = grid["z"][idx_floor[2, :]] + (idx[2, :] - idx_floor[2, :]) * grid["interval"]
    coordinate = np.vstack((x, y, z))

    return coordinate

def idx3d_flatten(idx3d, grid):
    '''
    The function to convert the 3D coordinate into 1D to index the column
    in grid["points"] more easily (and more closely to the original matlab implementation...)
    '''
    idx = idx3d[0] + grid["size"][0] * idx3d[1] + \
        grid["size"][0] * grid["size"][1] * idx3d[2]
    return idx

# Recover superquadric from SDF--------------------------------------------
def fitSuperquadricTSDF(sdf, x_init, truncation, points, roi_idx, boundingPoints, para):
    # grid is only used for debug visualization

    # TODO: convey these optimization options to the correct function
    # options = optimoptions(...
    #     'lsqnonlin', 'Algorithm', 'trust-region-reflective', ...
    #     'Display', 'off', 'MaxIterations', para.maxOptiIter);


    ## Format of x:
    # x = [e1, e2, ax, ay, az, r, p, y, tx, ty, tz]


    # Initialize validity vector
    valid = np.zeros(6)

    # Positional upper and lower bound
    t_lb = boundingPoints[:, 0].T
    t_ub = boundingPoints[:, 7].T

    # Setting upper and lower bound
    lb = np.concatenate((np.array([0.0, 0.0, truncation, truncation, truncation,\
        -2 * np.pi, -2 * np.pi, -2 * np.pi]), t_lb))
    ub = np.concatenate((np.array([2, 2, 1, 1, 1, \
        2 * np.pi, 2 * np.pi, 2 * np.pi]), t_ub))
    
    # Initialization
    x = x_init
    cost = 0
    switched = 0
    nan_idx = np.logical_not(np.isnan(sdf))
    sigma2 = np.exp(truncation) ** 2

    for iter in range(para.maxIter): 
        Rot = eul2rotm(x[5:8])
        checkPoints = np.array([x[8 : 11] - Rot[:, 0].T * x[2],
                    x[8 : 11] + Rot[:, 0].T * x[2],
                    x[8 : 11] - Rot[:, 1].T * x[3],
                    x[8 : 11] + Rot[:, 1].T * x[3],
                    x[8 : 11] - Rot[:, 2].T * x[4],
                    x[8 : 11] + Rot[:, 2].T * x[4]])
        valid[0 : 3] = (np.min(checkPoints, axis=0) >= t_lb - 1 * truncation)
        valid[3 : 6] = (np.max(checkPoints, axis=0) <= t_ub + 1 * truncation)

        # Break the iteration when validity is violated
        if not all(valid):
            break

        # calculating the signed distance of voxels to the current superquadric
        sdf_current = sdfSuperquadric(x, points, 0)

        # find the voxels activated by the current superquadric configuration
        active_idx = np.logical_and(np.logical_and(\
                        sdf_current < para.activeMultiplier* truncation,\
                        sdf_current > -para.activeMultiplier* truncation),\
                        nan_idx)       
        points_active = points[:, active_idx]
        sdf_active = sdf[active_idx]

        # Calculate the weight of each point according to difference
        weight = inlierWeight(sdf_active, active_idx, sdf_current, sigma2, para.w, truncation)

        # Scale upper bound
        Rot = eul2rotm(x[5:8])
        bP = boundingPoints - x[8 : 11].reshape(-1, 1)
        bP_body = np.matmul(Rot.T, bP) # Bounding points in body frame
        scale_limit = np.mean(abs(bP_body.T), axis=0)
        ub[2 : 5] = scale_limit

        try:
            optim_result = least_squares(differenceSQSDF, x, bounds=(lb, ub), \
                                        args=(sdf_active, points_active, truncation, weight),\
                                        max_nfev=para.maxOptiIter, jac='3-point')
            
            # TODO: Here, the different optimization libraries generate different results
            x_n = optim_result.x
            cost_n = optim_result.cost * 2 # Obey the Matlab convention...
        except ValueError: # If x0 is not in bound, np won't support it
            optim_result = least_squares(differenceSQSDF, (lb+ub)/2, bounds=(lb, ub), \
                                        args=(sdf_active, points_active, truncation, weight),\
                                        max_nfev=para.maxOptiIter, jac='3-point', xtol=1e-06)
            x_n = optim_result.x
            cost_n = optim_result.cost * 2 # Obey the Matlab convention...
        
        # update sigma 
        sigma2_n = cost_n / np.sum(weight)
        
        # average cost
        cost_n = cost_n / sdf_active.shape[0]
        
        # evaluate relative cost decrease
        relative_cost = abs(cost - cost_n) / cost_n
        
        if (cost_n < para.tolerance and iter > 0) or \
                (relative_cost < para.relative_tolerance and \
                    switched >= para.maxSwitch and iter > para.iter_min):
            # Break the loop and use the obtained parameters if several criteria are satisfied
            x = x_n
            break
        
        if relative_cost < para.switch_tolerance and iter != 0 and switched < para.maxSwitch:
        
            # activate switching algorithm to avoid local minimum
            switch_success = 0
            
            # case1 - axis-mismatch similarity
            axis_0 = eul2rotm(x[5:8])
            axis_1 = np.roll(axis_0, 2, axis=1)
            axis_2 = np.roll(axis_0, 1, axis=1)
            eul_1 = rotm2eul(axis_1)
            eul_2 = rotm2eul(axis_2)


            # Paper "Robust and Accurate Superquadric Recovery: a Probabilistic Approach"
            # explains why the order is swapped here
            x_axis = np.array([\
                [x[1], x[0], x[3], x[4], x[2], eul_1[0], eul_1[1], eul_1[2], x[8], x[9], x[10]],\
                [x[1], x[0], x[4], x[2], x[3], eul_2[0], eul_2[1], eul_2[2], x[8], x[9], x[10]]\
                    ])
            
            # case2 - duality similarity and combinations
            scale_ratio = np.roll(x[2:5], 2) / x[2:5]
            scale_idx = np.nonzero(np.logical_and(scale_ratio > 0.8, scale_ratio < 1.2))[0]
            x_rot = np.zeros((scale_idx.shape[0], 11))
            rot_idx = 0
            
            # Generate new sq's with similar shapes (similar in shape but distance in parameter)
            if 0 in scale_idx:
                eul_rot = rotm2eul(axis_0 * rotz(45))
                if x[1] <= 1:
                    new_xy_length = ((1 - np.sqrt(2)) * x[1] + np.sqrt(2)) * min(x[2], x[3])
                    x_rot[rot_idx, :] = np.array(\
                        [x[0], 2 - x[1], new_xy_length, new_xy_length, x[4], \
                         eul_rot[0], eul_rot[1], eul_rot[2], x[8], x[9], x[10]]
                    )
                else:
                    new_xy_length = ((np.sqrt(2)/2 - 1) * x[1] + 2 - np.sqrt(2)/2) * min(x[2], x[3])
                    x_rot[rot_idx, :] = np.array(\
                        [x[0], 2 - x[1], new_xy_length, new_xy_length, x[4], \
                         eul_rot[0], eul_rot[1], eul_rot[2], x[8], x[9], x[10]])
                rot_idx = rot_idx + 1
            if 1 in scale_idx:
                eul_rot = rotm2eul(axis_1 * rotz(45))
                if x[0] <= 1:
                    new_xy_length = ((1 - np.sqrt(2)) * x[0] + np.sqrt(2)) * min(x[3], x[4])
                    x_rot[rot_idx, :] = np.array(\
                        [x[1], 2 - x[0], new_xy_length, new_xy_length, x[2], \
                         eul_rot[0], eul_rot[1], eul_rot[2], x[8], x[9], x[10]]
                    )
                else:
                    new_xy_length = ((np.sqrt(2)/2 - 1) * x[0] + 2 - np.sqrt(2)/2) * min(x[3], x[4])
                    x_rot[rot_idx, :] = np.array(\
                        [x[1], 2 - x[0], new_xy_length, new_xy_length, x[2], \
                         eul_rot[0], eul_rot[1], eul_rot[2], x[8], x[9], x[10]])
                rot_idx = rot_idx + 1
            if 2 in scale_idx:
                eul_rot = rotm2eul(axis_2 * rotz(45))
                if x[0] <= 1:
                    new_xy_length = ((1 - np.sqrt(2)) * x[0] + np.sqrt(2)) * min(x[4], x[2])
                    x_rot[rot_idx, :] = np.array(\
                        [x[1], 2 - x[0], new_xy_length, new_xy_length, x[3], \
                         eul_rot[0], eul_rot[1], eul_rot[2], x[8], x[9], x[10]]
                    )
                else:
                    new_xy_length = ((np.sqrt(2)/2 - 1) * x[0] + 2 - np.sqrt(2)/2) * min(x[4], x[2])
                    x_rot[rot_idx, :] = np.array(\
                        [x[1], 2 - x[0], new_xy_length, new_xy_length, x[3], \
                         eul_rot[0], eul_rot[1], eul_rot[2], x[8], x[9], x[10]])
                rot_idx = rot_idx + 1

            # generate candidate configuration list with cost
            x_candidate = np.vstack((x_axis, x_rot))
            cost_candidate = cost_switched(\
                x_candidate, sdf_active, points_active, truncation, weight)
            
            idx_nan = np.nonzero(\
                np.logical_and(\
                    np.logical_not(np.isnan(cost_candidate)), \
                    np.logical_not(np.isinf(cost_candidate)))
            )[0]
            cost_candidate = cost_candidate[idx_nan]
            x_candidate = x_candidate[idx_nan, :]
            
            idx = np.argsort(cost_candidate)
            for i_candidate in range(idx.shape[0]):
                
                # scale upper  bound
                Rot = eul2rotm(x_candidate[idx[i_candidate], 5 : 8])
                bP = boundingPoints - x_candidate[idx[i_candidate], 8:11].reshape(-1,1)
                bP_body = np.matmul(Rot.T, bP)
                scale_limit = np.mean(abs(bP_body.T), axis=0)
                ub[2 : 5] = scale_limit       
                try:
                    optim_result_switched = least_squares(differenceSQSDF,x_candidate[idx[i_candidate], :], bounds=(lb, ub), \
                                        args=(sdf_active, points_active, truncation, weight),\
                                        max_nfev=para.maxOptiIter, jac='3-point', xtol=1e-06)
                    x_switch = optim_result_switched.x
                    cost_switch = optim_result_switched.cost * 2# Obey the Matlab convention
                except ValueError: # The function needs the initial value to be inside the bounds
                    optim_result_switched = least_squares(differenceSQSDF, (lb+ub)/2, bounds=(lb, ub), \
                                        args=(sdf_active, points_active, truncation, weight),\
                                        max_nfev=para.maxOptiIter, jac='3-point')
                    x_switch = optim_result.x
                    cost_switch = optim_result.cost * 2 # Obey the Matlab convention...
                
                if cost_switch / sdf_active.shape[0] < min(cost_n, cost):
                    x = x_switch
                    cost = cost_switch / sdf_active.shape[0]
                    sigma2 = cost_switch / np.sum(weight)
                    switch_success = 1
                    break
            
            if switch_success == 0:        
                cost = cost_n
                x = x_n
                sigma2 = sigma2_n
        
            switched = switched + 1
        else:
            cost = cost_n
            sigma2 = sigma2_n
            x = x_n

    # TODO: figure out what this part does...=> the meaning of each variable
    sdf_occ = sdfSuperquadric(x, points, 0)
    occ = sdf_occ < para.nanRange
    occ_idx = roi_idx[occ]
    occ_in = sdf_occ <= 0

    num_idx = np.zeros(3)
    num_idx[0] = np.sum(np.logical_or(sdf[occ_in]<= 0, np.isnan(sdf[occ_in])))
    num_idx[1]= np.sum(sdf[occ_in] > 0)
    num_idx[2] = np.sum(sdf[occ_in]<= 0)

    # final check size validity
    Rot = eul2rotm(x[5 : 8])
    checkPoints = np.array([x[8 : 11] - Rot[:, 0].T * x[2],
                    x[8 : 11] + Rot[:, 0].T * x[2],
                    x[8 : 11] - Rot[:, 1].T * x[3],
                    x[8 : 11] + Rot[:, 1].T * x[3],
                    x[8 : 11] - Rot[:, 2].T * x[4],
                    x[8 : 11] + Rot[:, 2].T * x[4]])

    valid[0 : 3] = (np.min(checkPoints, axis=0) >= t_lb - 1 * truncation)
    valid[3 : 6] = (np.max(checkPoints, axis=0) <= t_ub + 1 * truncation)
    return x, occ_idx, valid, num_idx
# def eul2rotm(rpy):
#     '''
#     The function to convert euler angle into rotation matrix
#     '''
#     r = R.from_euler('zyx', rpy, degrees=False)
#     return r.as_matrix()

def eul2rotm(eul):
    Rot = np.zeros((3,3))
    ct = np.cos(eul)
    st = np.sin(eul)
    Rot[0, 0] = ct[1] * ct[0]
    Rot[0, 1] = st[2] * st[1] * ct[0] - ct[2] * st[0]
    Rot[0, 2] = ct[2] * st[1] *ct[0] + st[2] * st[0]
    Rot[1, 0] = ct[1] * st[0]
    Rot[1, 1] = st[2] * st[1] * st[0] + ct[2]*ct[0]
    Rot[1, 2] = ct[2] * st[1]* st[0] - st[2]*ct[0]
    Rot[2, 0] = -st[1]
    Rot[2, 1] = st[2] * ct[1]
    Rot[2, 2] = ct[2] * ct[1]     

    return Rot              

def rotm2eul(rot):
    '''
    The function to convert rotation matrix to euler angle
    '''
    r = R.from_matrix(rot)
    return r.as_euler('zyx', degrees=False)

def rotz(degree):
    '''
    The function to obtain the rotation matrix around z-axis
    '''
    r = R.from_quat([0, 0, np.sin(degree * np.pi/180), np.cos(degree * np.pi/180)])
    return r.as_matrix()


def cost_switched(para, sdf, points, truncation, weight):
    '''
    The function to calculat the cost of switched parameters
    '''
    value = np.zeros(para.shape[0])
    for i in range(para.shape[0]):
        value[i] = np.linalg.norm(differenceSQSDF(para[i, :], sdf, points, truncation, weight))
    return value


def sdfSuperquadric(para, points, truncation):
    '''
    The function to calculate the sdf of points w.r.t the current sq
    '''
        
    Rot = eul2rotm(para[5:8])
    t = para[8:11]
    X = np.matmul(Rot.T, points) - np.matmul(Rot.T, t.T).reshape(-1, 1)
    
    r0 = np.linalg.norm(X, axis=0)

    term1 = X[0, :] / para[2]
    term2 = X[1, :] / para[3]
    term3 = X[2, :] / para[4]

    # The equation (1) (2) mentioned in the paper
    scale = (((term1 ** 2) ** (1 / para[1]) + (term2 ** 2) ** (1 / para[1])) ** (para[1] / para[0]) + \
        (term3 ** 2) **(1 / para[0])) ** (-para[0]/ 2)
    
    sdf = r0 * (1 - scale)
    
    if truncation != 0:
        sdf = np.minimum(np.maximum(sdf, -truncation), truncation)

    return sdf


def inlierWeight(sdf_active, active_idx, sdf_current, sigma2, w, truncation):
    '''
    The function to calculate the weight for the sdf_current to match sdf_active
    '''
        
    inIdx = sdf_active < 0.0 * truncation
    sdf_current = sdf_current[active_idx]
    
    const = w / ((1 - w) * (2 * np.pi * sigma2) ** (- 1 / 2) * 1 * truncation)
    dist_current = np.minimum(np.maximum(sdf_current[inIdx], -truncation), truncation) - sdf_active[inIdx]
    
    weight = np.ones(sdf_active.shape)
    p = np.exp(-1 / (2 * sigma2) * dist_current ** 2)
    p = p / (const + p)
    weight[inIdx] = p
    return weight


def differenceSQSDF(para, sdf, points, truncation, weight):
    '''
    The function calculate the weighted distance between the sdf_pred and sdf_gt
    Reurn: dist: the minimization goal
    '''
    
    Rot = eul2rotm(para[5:8])
    t = para[8:11]
    X = np.matmul(Rot.T, points) - np.matmul(Rot.T, t.T).reshape(-1, 1)
    
    r0 = np.linalg.norm(X, axis=0)

    term1 = X[0, :] / para[2]
    term2 = X[1, :] / para[3]
    term3 = X[2, :] / para[4]

    # The equation (1) (2) mentioned in the paper
    scale = (((term1 ** 2) ** (1 / para[1]) + (term2 ** 2) ** (1 / para[1])) ** (para[1] / para[0]) + \
        (term3 ** 2) **(1 / para[0])) ** (-para[0]/ 2)
    
    sdf_para = r0 * (1 - scale)
    
    if truncation != 0:
        sdf_para = np.minimum(np.maximum(sdf_para, -truncation), truncation)
    
    dist = (sdf_para - sdf) * weight ** (1 / 2)

    return dist  
    