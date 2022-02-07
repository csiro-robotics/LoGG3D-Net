import os
import sys
import glob
import random
import torch
import numpy as np
import logging
import json
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from utils.data_loaders.pointcloud_dataset import *
from utils.visualization.o3d_tools import *

class MulRanDataset(PointCloudDataset):
  r"""
  Generate single pointcloud frame from MulRan dataset. 
  """
  def __init__(self,
               phase,
               random_rotation=False,
               random_occlusion=False,
               random_scale=False,
               config=None):

    self.root = root = config.mulran_dir 
    self.pnv_prep = config.pnv_preprocessing
    self.gp_rem = config.gp_rem
    self.int_norm = config.mulran_normalize_intensity

    PointCloudDataset.__init__(self, phase, random_rotation, random_occlusion, random_scale, config)

    logging.info("Initializing MulRanDataset")
    logging.info(f"Loading the subset {phase} from {root}")

    sequences = config.mulran_data_split[phase]
    for drive_id in sequences:
      inames = self.get_all_scan_ids(drive_id)
      for query_id, start_time in enumerate(inames):
        self.files.append((drive_id,query_id))


  def get_all_scan_ids(self, drive_id):
    sequence_path = self.root + drive_id + '/Ouster/'
    fnames = sorted(glob.glob(os.path.join(sequence_path,'*.bin')))
    assert len(
        fnames) > 0, f"Make sure that the path {self.root} has drive id: {drive_id}"
    inames = [int(os.path.split(fname)[-1][:-4]) for fname in fnames]
    return inames

  def get_velodyne_fn(self, drive_id, query_id):
    sequence_path = self.root + drive_id + '/Ouster/'
    fname = sorted(glob.glob(os.path.join(sequence_path,'*.bin')))[query_id]
    return fname

  def get_pointcloud_tensor(self, drive_id, pc_id):
    fname = self.get_velodyne_fn(drive_id, pc_id)
    xyzr = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
    range = np.linalg.norm(xyzr[:,:3], axis=1)
    range_filter = np.logical_and(range>0.1, range<80)
    xyzr = xyzr[range_filter]
    if self.int_norm:
      xyzr[:, 3] = np.clip(xyzr[:, 3], 0, 1000) / 1000.0
    if self.gp_rem:
      not_ground_mask = np.ones(len(xyzr), np.bool)
      raw_pcd = make_open3d_point_cloud(xyzr[:,:3], color=None)
      _, inliers = raw_pcd.segment_plane(0.2, 3, 250)
      not_ground_mask[inliers] = 0
      xyzr = xyzr[not_ground_mask]

    if self.pnv_prep:
      xyzr = self.pnv_preprocessing(xyzr)
    if self.random_rotation:
      xyzr = self.random_rotate(xyzr)
    if self.random_occlusion:
      xyzr = self.occlude_scan(xyzr)
    if self.random_scale and random.random() < 0.95:
      scale = self.min_scale + \
          (self.max_scale - self.min_scale) * random.random()
      xyzr = scale * xyzr   
    
    return xyzr


  def __getitem__(self, idx): 
    drive_id = self.files[idx][0]
    t0= self.files[idx][1]
    xyz0_th = self.get_pointcloud_tensor(drive_id, t0)
    meta_info = {'drive': drive_id, 't0': t0}

    return (xyz0_th,
            meta_info)



#####################################################################################
# Load poses
#####################################################################################


def load_poses_from_csv(file_name):
    with open(file_name, newline='') as f:
        reader = csv.reader(f)
        data_poses=list(reader)
    
    transforms = []
    positions = []
    for cnt, line in enumerate(data_poses):
      line_f = [float(i) for i in line]
      P  = np.vstack((np.reshape(line_f[1:], (3,4)),[0,0,0,1]))
      transforms.append(P)
      positions.append([P[0, 3], P[1, 3], P[2, 3]])
    return np.asarray(transforms), np.asarray(positions)


#####################################################################################
# Load timestamps
#####################################################################################


def load_timestamps_csv(file_name):
    with open(file_name, newline='') as f:
        reader = csv.reader(f)
        data_poses=list(reader)
    data_poses_ts = np.asarray([float(t)/1e9 for t in np.asarray(data_poses)[:,0]])
    return data_poses_ts