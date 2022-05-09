import os
import sys
import glob
import random
import torch
import numpy as np
import logging
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from utils.data_loaders.mulran.mulran_dataset import *
from utils.o3d_tools import *
from utils.misc_utils import hashM

from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize

class MulRanSparseTupleDataset(MulRanTupleDataset):
  r"""
  Generate tuples (anchor, positives, negatives) using distance
  Optional other_neg for quadruplet loss. 
  Convert all to sparse tensors
  """

  def __init__(self,
               phase,
               random_rotation=False,
               random_occlusion=False,
               random_scale=False,
               config=None):

    MulRanTupleDataset.__init__(self, phase, random_rotation, random_occlusion, random_scale, config)
    
    self.voxel_size = config.voxel_size
    self.num_points = config.num_points
    self.phase = phase
    self.gp_rem = config.gp_rem
    self.int_norm = config.mulran_normalize_intensity

  def get_pointcloud_sparse_tensor(self, drive_id, pc_id):
    fname = self.get_velodyne_fn(drive_id, pc_id)
    xyzr = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
    range = np.linalg.norm(xyzr[:,:3], axis=1)
    range_filter = np.logical_and(range>0.1, range<80)
    xyzr = xyzr[range_filter]
    if self.int_norm:
      xyzr[:, 3] = np.clip(xyzr[:, 3], 0, 1000) / 1000.0
    if self.gp_rem:
      use_ransac = False
      if use_ransac:
        # self.timer.tic()
        not_ground_mask = np.ones(len(xyzr), np.bool)
        raw_pcd = make_open3d_point_cloud(xyzr[:,:3], color=None)
        _, inliers = raw_pcd.segment_plane(0.2, 3, 250)
        not_ground_mask[inliers] = 0
        xyzr = xyzr[not_ground_mask]
        # self.timer.toc()
        # print(self.timer.avg)
      else:
        xyzr = xyzr[xyzr[:,2] > -0.9]

    if self.random_rotation:
      xyzr = self.random_rotate(xyzr)
    if self.random_occlusion:
      xyzr = self.occlude_scan(xyzr)
    if self.random_scale and random.random() < 0.95:
      scale = self.min_scale + \
          (self.max_scale - self.min_scale) * random.random()
      xyzr = scale * xyzr

    pc_ = np.round(xyzr[:, :3] / self.voxel_size).astype(np.int32)
    pc_ -= pc_.min(0, keepdims=1)
    feat_ = xyzr

    _, inds, inverse_map = sparse_quantize(pc_,
                                              return_index=True,
                                              return_inverse=True)
    
    if 'train' in self.phase:
      if len(inds) > self.num_points:
        inds = np.random.choice(inds, self.num_points, replace=False)
    
    pc = pc_[inds]
    feat = feat_[inds]
    sparse_pc = SparseTensor(feat, pc)
    inverse_map = SparseTensor(inverse_map, pc_)

    # return {
    #   'sparse_pc': sparse_pc,
    #   'inverse_map': inverse_map, # This inverse map is useless becuase of random.choice() above. 
    # }
    return sparse_pc

  def __getitem__(self, idx):
    drive_id, query_id = self.files[idx][0], self.files[idx][1]
    positive_ids, negative_ids = self.files[idx][2], self.files[idx][3]

    sel_positive_ids = random.sample(positive_ids, self.positives_per_query)
    sel_negative_ids = random.sample(negative_ids, self.negatives_per_query)
    positives, negatives, other_neg = [], [], None

    query_th = self.get_pointcloud_sparse_tensor(drive_id, query_id)
    for sp_id in sel_positive_ids:
      positives.append(self.get_pointcloud_sparse_tensor(drive_id, sp_id))
    for sn_id in sel_negative_ids:
      negatives.append(self.get_pointcloud_sparse_tensor(drive_id, sn_id))

    meta_info = {'drive': drive_id, 'query_id': query_id}
    
    if not self.quadruplet:
      return {
        'query': query_th,
        'positives': positives,
        'negatives': negatives,
        'meta_info': meta_info
      }
    else: #For Quadruplet Loss
      other_neg_id = self.get_other_negative(drive_id, query_id, sel_positive_ids, sel_negative_ids) 
      other_neg_th = self.get_pointcloud_sparse_tensor(drive_id, other_neg_id) 
      return {
        'query': query_th,
        'positives': positives,
        'negatives': negatives,
        'other_neg': other_neg_th,
        'meta_info': meta_info,
      }


class MulRanPointSparseTupleDataset(MulRanSparseTupleDataset):
  r"""
  Generate tuples (anchor, positives, negatives) using distance
  Optional other_neg for quadruplet loss. 
  Convert all to sparse tensors
  Return additional Positive Point Pairs (for point-wise loss)
  """

  def __init__(self,
               phase,
               random_rotation=False,
               random_occlusion=False,
               random_scale=False,
               config=None):

    MulRanSparseTupleDataset.__init__(self, phase, random_rotation, random_occlusion, random_scale, config)
    
    self.voxel_size = config.voxel_size
    self.num_points = config.num_points
    self.phase = phase
    self.gp_rem = config.gp_rem
    self.int_norm = config.mulran_normalize_intensity

  def base_2_lidar(self, wTb):
    bTl = np.asarray([-0.999982947984152,  -0.005839838492430,   -0.000005225706031,  1.7042,
                    0.005839838483221,   -0.999982947996283,   0.000001775876813,   -0.0210,
                    -0.000005235987756,  0.000001745329252,    0.999999999984769 ,  1.8047,
                    0, 0, 0, 1]
                    ).reshape(4, 4)
    return wTb @ bTl

  def get_delta_pose(self, transforms):
    w_T_p1 = self.base_2_lidar(transforms[0])
    w_T_p2 = self.base_2_lidar(transforms[1])

    p1_T_w = np.linalg.inv(w_T_p1)
    p1_T_p2 = np.matmul(p1_T_w, w_T_p2)
    return p1_T_p2

  def get_gt_transforms(self, drive, indices=None, ext='.txt', return_all=False):
    poses_path = self.root + drive + '/scan_poses.csv'
    poses_full, _ = load_poses_from_csv(poses_path)
    return poses_full[indices]

  def generate_rand_negative_pairs(self, positive_pairs, hash_seed, N0, N1, N_neg=0):
    """
    Generate random negative pairs
    """
    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)
    if N_neg < 1:
      N_neg = positive_pairs.shape[0] * 2
    pos_keys = hashM(positive_pairs, hash_seed)

    neg_pairs = np.floor(np.random.rand(int(N_neg), 2) * np.array([[N0, N1]])).astype(
        np.int64)
    neg_keys = hashM(neg_pairs, hash_seed)
    mask = np.isin(neg_keys, pos_keys, assume_unique=False)
    return neg_pairs[np.logical_not(mask)]

  def get_sparse_pcd(self, drive_id, pc_id):
    fname = self.get_velodyne_fn(drive_id, pc_id)
    xyzr = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
    range = np.linalg.norm(xyzr[:,:3], axis=1)
    range_filter = np.logical_and(range>0.1, range<80)
    xyzr = xyzr[range_filter]
    if self.int_norm:
      xyzr[:, 3] = np.clip(xyzr[:, 3], 0, 1000) / 1000.0
    if self.gp_rem:
      use_ransac = False
      if use_ransac:
        # self.timer.tic()
        not_ground_mask = np.ones(len(xyzr), np.bool)
        raw_pcd = make_open3d_point_cloud(xyzr[:,:3], color=None)
        _, inliers = raw_pcd.segment_plane(0.2, 3, 250)
        not_ground_mask[inliers] = 0
        xyzr = xyzr[not_ground_mask]
        # self.timer.toc()
        # print(self.timer.avg)
      else:
        xyzr = xyzr[xyzr[:,2] > -0.9]
    
    xyzr_copy = copy.deepcopy(xyzr)
    if self.random_rotation:
      xyzr = self.random_rotate(xyzr)
    pc_ = np.round(xyzr[:, :3] / self.voxel_size).astype(np.int32)
    pc_ -= pc_.min(0, keepdims=1)
    feat_ = xyzr
    _, inds = sparse_quantize(pc_,
                                  return_index=True,
                                  return_inverse=False)
    if len(inds) > self.num_points:
        inds = np.random.choice(inds, self.num_points, replace=False)
    
    st = SparseTensor(feat_[inds], pc_[inds])
    pcd = make_open3d_point_cloud(xyzr_copy[inds][:,:3], color=None)
    # o3d.visualization.draw_geometries([pcd])
    return st, pcd

  def get_point_tuples(self, drive_id, query_id, pos_id):
    q_st, q_pcd = self.get_sparse_pcd(drive_id, query_id)
    p_st, p_pcd = self.get_sparse_pcd(drive_id, pos_id)
    N0, N1 = len(q_pcd.points), len(p_pcd.points)
    p_pcd_temp = copy.deepcopy(p_pcd)

    matching_search_voxel_size = min(self.voxel_size*1.5, 0.1)
    all_odometry = self.get_gt_transforms(drive_id, [query_id, pos_id])
    delta_T = self.get_delta_pose(all_odometry)
    p_pcd.transform(delta_T)
    # draw_registration_result(q_pcd, p_pcd, np.eye(4))
    reg = o3d.pipelines.registration.registration_icp(
            p_pcd, q_pcd, 0.2, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
    p_pcd.transform(reg.transformation)
    # draw_registration_result(q_pcd, p_pcd, np.eye(4))

    pos_pairs = get_matching_indices(q_pcd, p_pcd, matching_search_voxel_size)
    assert pos_pairs.ndim ==2, f"No pos_pairs for {query_id} in drive id: {drive_id}"
    # neg_pairs = self.generate_rand_negative_pairs(pos_pairs, max(N0, N1), N0, N1)

    dist_debug = False
    if dist_debug:
      # print(reg.transformation)
      delta_T = reg.transformation @ delta_T
      # draw_registration_result(q_pcd, p_pcd, np.eye(4))
      draw_registration_result(q_pcd, p_pcd_temp, delta_T)
      q_pcd_xyz, p_pcd_xyz = np.asarray(q_pcd.points), np.asarray(p_pcd.points)
      pos0 = q_pcd_xyz[pos_pairs[:, 0]]
      pos1 = p_pcd_xyz[pos_pairs[:, 1]]
      # neg0 = q_pcd_xyz[neg_pairs[:, 0]]
      # neg1 = p_pcd_xyz[neg_pairs[:, 1]]
      pos_point_dists = np.sum(np.power((pos0 - pos1), 2), axis=1)
      print(np.min(pos_point_dists), np.max(pos_point_dists))
      # neg_point_dists = np.sum(np.power((neg0 - neg1), 2), axis=1)

    return q_st, p_st, pos_pairs#, neg_pairs

  def __getitem__(self, idx):
    drive_id, query_id = self.files[idx][0], self.files[idx][1]
    positive_ids, negative_ids = self.files[idx][2], self.files[idx][3]

    sel_positive_ids = random.sample(positive_ids, self.positives_per_query)
    sel_negative_ids = random.sample(negative_ids, self.negatives_per_query)
    positives, negatives, other_neg = [], [], None

    query_st, p_st, pos_pairs = self.get_point_tuples(drive_id, query_id, sel_positive_ids[0])
    positives.append(p_st)

    for sp_id in sel_positive_ids[1:]:
      positives.append(self.get_pointcloud_sparse_tensor(drive_id, sp_id))
    for sn_id in sel_negative_ids:
      negatives.append(self.get_pointcloud_sparse_tensor(drive_id, sn_id))

    meta_info = {'drive': drive_id, 'query_id': query_id, 'pos_pairs': pos_pairs}

    if not self.quadruplet:
      return {
        'query': query_st,
        'positives': positives,
        'negatives': negatives,
        'meta_info': meta_info
      }
    else: #For Quadruplet Loss
      other_neg_id = self.get_other_negative(drive_id, query_id, sel_positive_ids, sel_negative_ids) 
      other_neg_st = self.get_pointcloud_sparse_tensor(drive_id, other_neg_id) 
      return {
        'query': query_st,
        'positives': positives,
        'negatives': negatives,
        'other_neg': other_neg_st,
        'meta_info': meta_info,
      }
