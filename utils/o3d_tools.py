# Some of these functions are copied from: https://github.com/chrischoy/FCGF

import copy
import numpy as np
import open3d as o3d


def visualize_scan_open3d(ptcloud_xyz, colors=[]):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ptcloud_xyz)
    if colors != []:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


def make_open3d_point_cloud(xyz, color=None, tile=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        if tile:
            if len(color) != len(xyz):
                color = np.tile(color, (len(xyz), 1))
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def get_matching_indices(source, target, search_voxel_size, K=None):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    pcd_tree = o3d.geometry.KDTreeFlann(target_copy)

    match_inds = []
    for i, point in enumerate(source_copy.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(
            point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append([i, j])
    return np.asarray(match_inds)


def downsample_point_cloud(xyzr, voxel_size=0.05):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzr[:, :3])
    pcd_ds, ds_trace, ds_ids = pcd.voxel_down_sample_and_trace(
        voxel_size, pcd.get_min_bound(), pcd.get_max_bound(), False)
    inv_ids = [ids[0] for ids in ds_ids]
    ds_intensities = np.asarray(xyzr[:, 3])[inv_ids]
    return np.hstack((pcd_ds.points, ds_intensities.reshape(-1, 1)))


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    target_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
