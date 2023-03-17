import os
import sys
import glob
import random
import numpy as np
import logging
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from utils.misc_utils import Timer
from utils.o3d_tools import *
from utils.data_loaders.pointcloud_dataset import *

class KittiDataset(PointCloudDataset):
    r"""
    Generate single pointcloud frame from KITTI odometry dataset. 
    """

    def __init__(self,
                 phase,
                 random_rotation=False,
                 random_occlusion=False,
                 random_scale=False,
                 config=None):

        self.root = root = config.kitti_dir
        self.gp_rem = config.gp_rem
        self.pnv_prep = config.pnv_preprocessing
        self.timer = Timer()

        PointCloudDataset.__init__(
            self, phase, random_rotation, random_occlusion, random_scale, config)

        logging.info("Initializing KittiDataset")
        logging.info(f"Loading the subset {phase} from {root}")
        if self.gp_rem:
            logging.info("Dataloader initialized with Ground Plane removal.")

        sequences = config.kitti_data_split[phase]
        for drive_id in sequences:
            drive_id = int(drive_id)
            inames = self.get_all_scan_ids(drive_id, is_sorted=True)
            for start_time in inames:
                self.files.append((drive_id, start_time))

    def get_all_scan_ids(self, drive_id, is_sorted=False):
        fnames = glob.glob(
            self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
        assert len(
            fnames) > 0, f"Make sure that the path {self.root} has drive id: {drive_id}"
        inames = [int(os.path.split(fname)[-1][:-4]) for fname in fnames]
        if is_sorted:
            return sorted(inames)
        return inames

    def get_velodyne_fn(self, drive, t):
        fname = self.root + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
        return fname

    def get_pointcloud_tensor(self, drive_id, pc_id):
        fname = self.get_velodyne_fn(drive_id, pc_id)
        xyzr = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)

        if self.gp_rem:
            not_ground_mask = np.ones(len(xyzr), bool)
            raw_pcd = make_open3d_point_cloud(xyzr[:, :3], color=None)
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
        t0 = self.files[idx][1]

        xyz0_th = self.get_pointcloud_tensor(drive_id, t0)
        meta_info = {'drive': drive_id, 't0': t0}

        return (xyz0_th,
                meta_info)


class KittiTupleDataset(KittiDataset):
    r"""
    Generate tuples (anchor, positives, negatives) using distance
    Optional other_neg for quadruplet loss. 
    """

    def __init__(self,
                 phase,
                 random_rotation=False,
                 random_occlusion=False,
                 random_scale=False,
                 config=None):
        self.root = root = config.kitti_dir
        self.positives_per_query = config.positives_per_query
        self.negatives_per_query = config.negatives_per_query
        self.quadruplet = False
        self.gp_rem = config.gp_rem
        self.pnv_prep = config.pnv_preprocessing
        if config.train_loss_function == 'quadruplet':
            self.quadruplet = True

        PointCloudDataset.__init__(
            self, phase, random_rotation, random_occlusion, random_scale, config)

        logging.info("Initializing KittiTupleDataset")
        logging.info(f"Loading the subset {phase} from {root}")

        sequences = config.kitti_data_split[phase]
        tuple_dir = os.path.join(os.path.dirname(
            __file__), '../../../config/kitti_tuples/')
        self.dict_3m = json.load(open(tuple_dir + config.kitti_3m_json, "r"))
        self.dict_20m = json.load(open(tuple_dir + config.kitti_20m_json, "r"))
        self.kitti_seq_lens = config.kitti_seq_lens
        for drive_id in sequences:
            drive_id = int(drive_id)
            fnames = glob.glob(
                root + '/sequences/%02d/velodyne/*.bin' % drive_id)
            assert len(
                fnames) > 0, f"Make sure that the path {root} has data {drive_id}"
            inames = sorted([int(os.path.split(fname)[-1][:-4])
                            for fname in fnames])

            for query_id in inames:
                positives = self.get_positives(drive_id, query_id)
                negatives = self.get_negatives(drive_id, query_id)
                self.files.append((drive_id, query_id, positives, negatives))

    def get_positives(self, sq, index):
        sq = str(int(sq))
        assert sq in self.dict_3m.keys(), f"Error: Sequence {sq} not in json."
        sq_1 = self.dict_3m[sq]
        if str(int(index)) in sq_1:
            positives = sq_1[str(int(index))]
        else:
            positives = []
        if index in positives:
            positives.remove(index)
        return positives

    def get_negatives(self, sq, index):
        sq = str(int(sq))
        assert sq in self.dict_20m.keys(), f"Error: Sequence {sq} not in json."
        sq_2 = self.dict_20m[sq]
        all_ids = set(np.arange(self.kitti_seq_lens[sq]))
        neg_set_inv = sq_2[str(int(index))]
        neg_set = all_ids.difference(neg_set_inv)
        negatives = list(neg_set)
        if index in negatives:
            negatives.remove(index)
        return negatives

    def get_other_negative(self, drive_id, query_id, sel_positive_ids, sel_negative_ids):
        # Dissimillar to all pointclouds in triplet tuple.
        all_ids = range(self.kitti_seq_lens[str(drive_id)])
        neighbour_ids = sel_positive_ids
        for neg in sel_negative_ids:
            neg_postives_files = self.get_positives(drive_id, neg)
            for pos in neg_postives_files:
                neighbour_ids.append(pos)
        possible_negs = list(set(all_ids) - set(neighbour_ids))
        assert len(
            possible_negs) > 0, f"No other negatives for drive {drive_id} id {query_id}"
        other_neg_id = random.sample(possible_negs, 1)
        return other_neg_id[0]

    def __getitem__(self, idx):
        drive_id, query_id = self.files[idx][0], self.files[idx][1]
        positive_ids, negative_ids = self.files[idx][2], self.files[idx][3]

        sel_positive_ids = random.sample(
            positive_ids, self.positives_per_query)
        sel_negative_ids = random.sample(
            negative_ids, self.negatives_per_query)
        positives, negatives, other_neg = [], [], None

        query_th = self.get_pointcloud_tensor(drive_id, query_id)
        for sp_id in sel_positive_ids:
            positives.append(self.get_pointcloud_tensor(drive_id, sp_id))
        for sn_id in sel_negative_ids:
            negatives.append(self.get_pointcloud_tensor(drive_id, sn_id))

        meta_info = {'drive': drive_id, 'query_id': query_id}

        if not self.quadruplet:
            return (query_th,
                    positives,
                    negatives,
                    meta_info)
        else:  # For Quadruplet Loss
            other_neg_id = self.get_other_negative(
                drive_id, query_id, sel_positive_ids, sel_negative_ids)
            other_neg_th = self.get_pointcloud_tensor(drive_id, other_neg_id)
            return (query_th,
                    positives,
                    negatives,
                    other_neg_th,
                    meta_info)


#####################################################################################
# Load poses
#####################################################################################

def transfrom_cam2velo(Tcam):
    R = np.array([7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                  -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
                  ]).reshape(3, 3)
    t = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
    cam2velo = np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

    return Tcam @ cam2velo


def load_poses_from_txt(file_name):
    """
    Modified function from: https://github.com/Huangying-Zhan/kitti-odom-eval/blob/master/kitti_odometry.py
    """
    f = open(file_name, 'r')
    s = f.readlines()
    f.close()
    transforms = {}
    positions = []
    for cnt, line in enumerate(s):
        P = np.eye(4)
        line_split = [float(i) for i in line.split(" ") if i != ""]
        withIdx = len(line_split) == 13
        for row in range(3):
            for col in range(4):
                P[row, col] = line_split[row*4 + col + withIdx]
        if withIdx:
            frame_idx = line_split[0]
        else:
            frame_idx = cnt
        transforms[frame_idx] = transfrom_cam2velo(P)
        positions.append([P[0, 3], P[2, 3], P[1, 3]])
    return transforms, np.asarray(positions)


#####################################################################################
# Load timestamps
#####################################################################################


def load_timestamps(file_name):
    # file_name = data_dir + '/times.txt'
    file1 = open(file_name, 'r+')
    stimes_list = file1.readlines()
    s_exp_list = np.asarray([float(t[-4:-1]) for t in stimes_list])
    times_list = np.asarray([float(t[:-2]) for t in stimes_list])
    times_listn = [times_list[t] * (10**(s_exp_list[t]))
                   for t in range(len(times_list))]
    file1.close()
    return times_listn
