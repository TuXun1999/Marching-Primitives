import numpy as np
import sys
import open3d as o3d
import csv
from src.MPS import MPS, eul2rotm, parseInputArgs
from superquadrics import create_superellipsoids
import scipy.io
import cupy as cp

# Loading file paths
file_path = "../data/chair11_normalized.csv" # Specify the location of the csv file

# Read the csv file & Extract out SDF
sdf = []
with open(file_path, newline='') as csvfile:
    sdf_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in sdf_reader:
        sdf.append(float(row[0]))
sdf = np.array(sdf)
print(sdf[0])
# sdf = sdf.T

# Build up the voxel grid
voxelGrid = {}
voxelGrid['size'] = (np.ones(3) * sdf[0]).astype(int)
voxelGrid['range'] = sdf[1:7]
sdf = sdf[7:]

voxelGrid['x'] = np.linspace(voxelGrid['range'][0], voxelGrid['range'][1], voxelGrid['size'][0])
voxelGrid['y'] = np.linspace(voxelGrid['range'][2], voxelGrid['range'][3], voxelGrid['size'][1])
voxelGrid['z'] = np.linspace(voxelGrid['range'][4], voxelGrid['range'][5], voxelGrid['size'][2])
x, y, z = np.meshgrid(voxelGrid['x'], voxelGrid['y'], voxelGrid['z'])

# Permute the order
x = np.transpose(x, (1, 0, 2))
y = np.transpose(y, (1, 0, 2))
z = np.transpose(z, (1, 0, 2))

# Construct the points in voxelGrid (obey the convention in original Matlab codes)
s = np.stack([x, y, z], axis=3)
s = s.reshape(-1, 3, order='F').T
voxelGrid['points'] = s

voxelGrid['interval'] = (voxelGrid['range'][1] - voxelGrid['range'][0]) / (voxelGrid['size'][0] - 1)
voxelGrid['truncation'] = 1.2 * voxelGrid['interval']
voxelGrid['disp_range'] = [-np.inf, voxelGrid['truncation']]
voxelGrid['visualizeArclength'] = 0.01 * np.sqrt(voxelGrid['range'][1] - voxelGrid['range'][0])

sdf = np.clip(sdf, -voxelGrid['truncation'], voxelGrid['truncation'])

# marching-primitives
import time
start_time = time.time()
# Parsing varargin
para = parseInputArgs(voxelGrid, sys.argv[1:])
x = MPS(sdf, voxelGrid, para) 

# TODO: correct the codes
# TODO: Temporarily, use the data from Matlab directly
# x = scipy.io.loadmat('matlab_chair11.mat').get('x')
print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
# triangularization and compression
# mesh_original = meshSuperquadrics(x, 'Arclength', voxelGrid['visualizeArclength'])
# mesh = reducepatch(mesh_original.faces, mesh_original.vertices, 0.1)

# Read the original mesh file
mesh_filename = "../data/chair11.obj"

# Read the file as a triangular mesh
mesh = o3d.io.read_triangle_mesh(mesh_filename)
# Normalize mesh
mesh_scale = 0.8
vertices = np.asarray(mesh.vertices)
bbmin = np.min(vertices, axis=0)
bbmax = np.max(vertices, axis=0)
center = (bbmin + bbmax) * 0.5
scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
vertices = (vertices - center) * scale
mesh.vertices = o3d.utility.Vector3dVector(vertices)

sq_mesh = []
for i in range(x.shape[0]):
    e1, e2, a1, a2, a3, r, p, y, t1, t2, t3 = x[i, :]
    if e1 < 0.01:
        e1 = 0.01
    if e2 < 0.01:
        e2 = 0.01
    sq_vertices = create_superellipsoids(e1, e2, a1, a2, a3)
    rot = eul2rotm(np.array([r, p, y]))
    sq_vertices = np.matmul(rot, sq_vertices.T).T + np.array([t1, t2, t3])

    # Construct a point cloud representing the reconstructed object mesh
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sq_vertices)
    # Visualize the super-ellipsoids
    pcd.paint_uniform_color((0.0, 0.4, 0.0))

    sq_mesh.append(pcd)
# Create the window to display everything
vis= o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(mesh)
for val in sq_mesh:
    vis.add_geometry(val)
vis.run()

# Close all windows
vis.destroy_window()
'''
This is the part to save the mesh into the format of mat & stl
TODO: after correction of the corresponding steps
'''
# stl = mesh.to_stl()

# ifsave = True
# pathname, name, ext = os.path.splitext(os.path.join(file_path, file))
# if ifsave:
#     savemat(os.path.join(pathname, f"{name}_sq.mat"), {'x_save': x.astype(np.float32)})
#     stl.save(os.path.join(pathname, f"{name}_sq.stl"), mode=stl.Mode.BINARY)

'''
Visualization codes from the original Matlab file
TODO: should not include this part; we should have
our own ways to visualize the sq's (especially in mesh_original)
'''
# # visualize
# plt.close('all')
# view_vector = [151, -40]
# light_vector = [190, 10]
# camera_roll = 50
# color = [145, 163, 176] / 255

# sdf3d_region = sdf.reshape(voxelGrid['size'])
# sdf3d = np.transpose(sdf3d_region, (1, 0, 2))

# plydata = PlyData.read(os.path.join(pathname, f"{name}_watertight.ply"))
# mesh_gt = {'f': plydata.elements[0].data['vertex_indices'], 'v': np.array([plydata.elements[0].data['x'], plydata.elements[0].data['y'], plydata.elements[0].data['z']]).T}

# fig1 = plt.figure(1)
# ax1 = fig1.add_subplot(111, projection='3d')
# ax1.plot_trisurf(mesh_gt['v'][:, 0], mesh_gt['v'][:, 1], mesh_gt['v'][:, 2], triangles=mesh_gt['f'], color=color, alpha=1, edgecolor='none')
# ax1.set_aspect('equal')
# ax1.view_init(view_vector[1], view_vector[0])
# ax1.roll(camera_roll)
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# ax1.set_zlabel('Z')
# ax1.set_title('ground truth mesh from marching cubes')

# fig2 = plt.figure(2)
# ax2 = fig2.add_subplot(111, projection='3d')
# ax2.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], triangles=mesh.faces, color=color, alpha=1, edgecolor='none')
# ax2.set_aspect('equal')
# ax2.view_init(view_vector[1], view_vector[0])
# ax2.roll(camera_roll)
# ax2.set_xlabel('X')
# ax2.set_ylabel('Y')
# ax2.set_zlabel('Z')
# ax2.set_title('superquadrics representation from marching primitives')

# fig3 = plt.figure(3)
# ax3 = fig3.add_subplot(111, projection='3d')
# ax3.plot_trisurf(mesh_gt['v'][:, 0], mesh_gt['v'][:, 1], mesh_gt['v'][:, 2], triangles=mesh_gt['f'], color='g', alpha=0.5, edgecolor='none')
# ax3.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], triangles=mesh.faces, color=color, alpha=1, edgecolor='none')
# ax3.set_aspect('equal')
# ax3.view_init(view_vector[1], view_vector[0])
# ax3.roll(camera_roll)
# ax3.set_xlabel('X')
# ax3.set_ylabel('Y')
# ax3.set_zlabel('Z')
# ax3.set_title('overlapping recovered representation with the ground truth')

# plt.show()

