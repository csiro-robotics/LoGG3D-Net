import copy
import numpy as np
import open3d as o3d

def visualize_scan_open3d(ptcloud_xyz, colors = []):
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

def downsample_point_cloud(xyzr, voxel_size=0.05):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(xyzr[:,:3])
  pcd_ds, ds_trace, ds_ids = pcd.voxel_down_sample_and_trace(voxel_size, pcd.get_min_bound(), pcd.get_max_bound(), False)
  inv_ids = [ids[0] for ids in ds_ids]
  # inv_ids2 = [next(obj for obj in tr if obj != -1) for tr in ds_trace]
  ds_intensities = np.asarray(xyzr[:, 3])[inv_ids]
  return np.hstack((pcd_ds.points, ds_intensities.reshape(-1,1)))