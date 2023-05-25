import os
import sys
import random
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from torchsparse.utils.quantize import sparse_quantize
from torchsparse import SparseTensor
from utils.misc_utils import hashM
from utils.o3d_tools import *
from utils.data_loaders.general.general_dataset import *
from scipy.spatial.transform import Rotation as R
class GeneralSparseTupleDataset(GeneralTupleDataset):
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

        GeneralTupleDataset.__init__(
            self, phase, random_rotation, random_occlusion, random_scale, config)

        self.voxel_size = config.voxel_size
        self.num_points = config.num_points
        self.phase = phase
        self.downsample = config.downsample
        self.gp_rem = config.gp_rem
        self.gp_vals = config.gp_vals
        self.int_norm = config.mulran_normalize_intensity

    def get_pointcloud_sparse_tensor(self,  base_dir, rel_path, dataset, get_pcd=False):
        fname = os.path.join(base_dir, rel_path)
        if dataset == 'ugv' or dataset == 'apollo' or dataset == 'bushwalk':
            pcd = o3d.io.read_point_cloud(fname) # TODO: add numpy load, conditional
            
        if dataset == 'mulran' or dataset == 'kitti':
            xyzr = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
            pcd = make_open3d_point_cloud(xyzr[:, :3], color=None)
            
        # downpcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        xyz = np.asarray(pcd.points)
        oo = np.ones(len(xyz)).reshape((-1,1))
        xyzr = np.hstack((xyz, oo)).astype(np.float32)

        range = np.linalg.norm(xyzr[:, :3], axis=1)
        range_filter = np.logical_and(range > 0.1, range < 180)
        xyzr = xyzr[range_filter]

        if self.gp_rem:
            gp_val = self.gp_vals[dataset]
            xyzr = xyzr[xyzr[:,2] > - gp_val]

        xyzr_copy = copy.deepcopy(xyzr)
        # if self.pnv_prep:
        #     xyzr = self.pnv_preprocessing(xyzr)
        if self.random_rotation:
            xyzr = self.random_rotate(xyzr)
        # if self.random_occlusion:
        #     xyzr = self.occlude_scan(xyzr)
        # if self.random_scale and random.random() < 0.95:
        #     scale = self.min_scale + \
        #         (self.max_scale - self.min_scale) * random.random()
        #     xyzr = scale * xyzr


        pc_ = np.round(xyzr[:, :3] / self.voxel_size).astype(np.int32)
        pc_ -= pc_.min(0, keepdims=1)
        feat_ = xyzr

        _, inds = sparse_quantize(pc_,
                                    return_index=True,
                                    return_inverse=False)

        if 'train' in self.phase:
            if len(inds) > self.num_points:
                inds = np.random.choice(inds, self.num_points, replace=False)

        st = SparseTensor(feat_[inds], pc_[inds])
        if get_pcd:
            pcd = make_open3d_point_cloud(xyzr_copy[inds][:, :3], color=None)
            return st, pcd
        else:
            return st


    def __getitem__(self, idx):
        anchor_data = self.files[idx]
        positive_ids, non_negative_ids = anchor_data.positives, anchor_data.non_negatives
        negative_ids = self.get_negatives(non_negative_ids)

        sel_positive_ids = random.sample(
            list(positive_ids), self.positives_per_query)
        sel_negative_ids = random.sample(
            list(negative_ids), self.negatives_per_query)
        positives, negatives, other_neg = [], [], None

        query_th = self.get_pointcloud_sparse_tensor(self.base_dirs[anchor_data.dataset], anchor_data.rel_scan_filepath, anchor_data.dataset)
        for sp_id in sel_positive_ids:
            sp_data = self.files[sp_id]
            positives.append(self.get_pointcloud_sparse_tensor(self.base_dirs[anchor_data.dataset], sp_data.rel_scan_filepath, anchor_data.dataset))
        for sn_id in sel_negative_ids:
            sn_data = self.files[sn_id]
            negatives.append(self.get_pointcloud_sparse_tensor(self.base_dirs[anchor_data.dataset], sn_data.rel_scan_filepath, anchor_data.dataset))

        meta_info = {'drive': anchor_data.dataset, 'query_id': anchor_data.id}

        if not self.quadruplet:
            return (query_th,
                    positives,
                    negatives,
                    meta_info)
        else:  # For Quadruplet Loss
            other_neg_id = self.get_other_negative(anchor_data.dataset, idx, sel_positive_ids, sel_negative_ids)
            other_neg_data = self.files[other_neg_id]
            other_neg_th = self.get_pointcloud_sparse_tensor(self.base_dirs[anchor_data.dataset], other_neg_data.rel_scan_filepath, anchor_data.dataset)
            return (query_th,
                    positives,
                    negatives,
                    other_neg_th,
                    meta_info)


class GeneralPointSparseTupleDataset(GeneralSparseTupleDataset):
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

        GeneralSparseTupleDataset.__init__(
            self, phase, random_rotation, random_occlusion, random_scale, config)

        self.voxel_size = config.voxel_size
        self.num_points = config.num_points
        self.phase = phase
        self.downsample = config.downsample
        self.gp_rem = config.gp_rem
        self.gp_vals = config.gp_vals
        self.int_norm = config.mulran_normalize_intensity

    def mulran_base_2_lidar(self, wTb):
        bTl = np.asarray([-0.999982947984152,  -0.005839838492430,   -0.000005225706031,  1.7042,
                          0.005839838483221,   -0.999982947996283,   0.000001775876813,   -0.0210,
                          -0.000005235987756,  0.000001745329252,    0.999999999984769,  1.8047,
                          0, 0, 0, 1]
                         ).reshape(4, 4)
        return wTb @ bTl
    
    def pose_to_transform(self, pose):
        transform = np.eye(4)
        # T[:3, :3] = R.from_quat([pose[5], pose[6], pose[7], pose[4]]).as_matrix()
        transform[:3, :3] = R.from_quat([pose[3], pose[4], pose[5], pose[6]]).as_matrix()
        transform[:3,3] = pose[:3]
        return transform


    def get_delta_pose(self, drive_id, transforms):
        w_T_p1 = transforms[0]
        w_T_p2 = transforms[1]
        if drive_id == 'mulran':
            w_T_p1 = self.mulran_base_2_lidar(w_T_p1)
            w_T_p2 = self.mulran_base_2_lidar(w_T_p2)
        if drive_id == 'bushwalk':
            w_T_p1 = self.pose_to_transform(w_T_p1)
            w_T_p2 = self.pose_to_transform(w_T_p2)

        p1_T_w = np.linalg.inv(w_T_p1)
        p1_T_p2 = np.matmul(p1_T_w, w_T_p2)
        return p1_T_p2

    def get_point_tuples(self, drive_id, query_id, pos_id):
        drive_path = self.base_dirs[drive_id]
        query_data = self.files[query_id]
        pos_data = self.files[pos_id]
        q_st, q_pcd = self.get_pointcloud_sparse_tensor(drive_path, query_data.rel_scan_filepath, query_data.dataset, get_pcd=True)
        p_st, p_pcd = self.get_pointcloud_sparse_tensor(drive_path, pos_data.rel_scan_filepath, pos_data.dataset, get_pcd=True)

        matching_search_voxel_size = min(self.voxel_size*1.5, 0.1)
        # all_odometry = self.get_gt_transforms(drive_id, [query_id, pos_id])
        all_odometry = [query_data.pose, pos_data.pose]
        delta_T = self.get_delta_pose(drive_id, all_odometry)
        p_pcd.transform(delta_T)
        # draw_registration_result(q_pcd, p_pcd)

        reg = o3d.pipelines.registration.registration_icp(
            p_pcd, q_pcd, 0.2, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
        p_pcd.transform(reg.transformation)
        # draw_registration_result(q_pcd, p_pcd)

        pos_pairs = get_matching_indices(
            q_pcd, p_pcd, matching_search_voxel_size)
        # assert pos_pairs.ndim == 2, f"No pos_pairs for {query_id} in drive id: {drive_id}"

        return q_st, p_st, pos_pairs

    def __getitem__(self, idx):
        anchor_data = self.files[idx]
        positive_ids, non_negative_ids = anchor_data.positives, anchor_data.non_negatives
        negative_ids = self.get_negatives(idx, non_negative_ids)

        pos_samples = min(len(positive_ids), self.positives_per_query)
        sel_positive_ids = random.sample(
            list(positive_ids), pos_samples)
        sel_negative_ids = random.sample(
            list(negative_ids), self.negatives_per_query)
        positives, negatives, other_neg = [], [], None

        # print('\n', anchor_data.dataset, idx)
        query_st, p_st, pos_pairs = self.get_point_tuples(
            anchor_data.dataset, idx, sel_positive_ids[0])
        positives.append(p_st)


        for sp_id in sel_positive_ids[1:]:
            sp_data = self.files[sp_id]
            positives.append(self.get_pointcloud_sparse_tensor(self.base_dirs[anchor_data.dataset], sp_data.rel_scan_filepath, anchor_data.dataset))
        for sn_id in sel_negative_ids:
            sn_data = self.files[sn_id]
            negatives.append(self.get_pointcloud_sparse_tensor(self.base_dirs[anchor_data.dataset], sn_data.rel_scan_filepath, anchor_data.dataset))

        meta_info = {'drive': anchor_data.dataset, 'query_id': anchor_data.id, 'pos_pairs': pos_pairs}

        if not self.quadruplet:
            return {
                'query': query_st,
                'positives': positives,
                'negatives': negatives,
                'meta_info': meta_info
            }
        else:  # For Quadruplet Loss
            other_neg_id = self.get_other_negative(anchor_data.dataset, idx, sel_positive_ids, sel_negative_ids)
            other_neg_data = self.files[other_neg_id]
            other_neg_st = self.get_pointcloud_sparse_tensor(self.base_dirs[anchor_data.dataset], other_neg_data.rel_scan_filepath, anchor_data.dataset)
            return {
                'query': query_st,
                'positives': positives,
                'negatives': negatives,
                'other_neg': other_neg_st,
                'meta_info': meta_info,
            }
