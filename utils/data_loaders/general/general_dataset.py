import os
import sys
import random
import numpy as np
import logging
import open3d as o3d
import pickle
from typing import List, Dict

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from utils.o3d_tools import *
from utils.data_loaders.pointcloud_dataset import *

class TrainingTuple:
    # Tuple describing an element for training/validation
    def __init__(self, id: int, timestamp: int, rel_scan_filepath: str, positives: np.ndarray,
                 non_negatives: np.ndarray, pose: np, positives_poses: Dict[int, np.ndarray] = None):
        # id: element id (ids start from 0 and are consecutive numbers)
        # ts: timestamp
        # rel_scan_filepath: relative path to the scan
        # positives: sorted ndarray of positive elements id
        # negatives: sorted ndarray of elements id
        # pose: pose as 4x4 matrix
        # positives_poses: relative poses of positive examples refined using ICP
        self.id = id
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.positives = positives
        self.non_negatives = non_negatives
        self.pose = pose
        self.positives_poses = positives_poses


class GeneralTrainingTuple:
    # Tuple describing an element for training/validation
    def __init__(self, dataset: str, training_tuple : TrainingTuple, id_offset: int=0):
        self.dataset = dataset
        self.id = training_tuple.id + id_offset
        self.timestamp = training_tuple.timestamp
        self.rel_scan_filepath = training_tuple.rel_scan_filepath
        self.positives = np.asarray([ p + id_offset for p in training_tuple.positives])
        self.non_negatives = np.asarray([ p + id_offset for p in training_tuple.non_negatives])
        self.pose = training_tuple.pose
        # self.positives_poses = training_tuple.positives_poses

class GeneralDatasetEval(PointCloudDataset):
    r"""
    Generate single pointcloud frame from General dataset. 
    """

    def __init__(self,
                 phase,
                 random_rotation=False,
                 random_occlusion=False,
                 random_scale=False,
                 config=None):

        # self.root = root = config.general_dir
        self.base_dirs = {"kitti" : config.kitti_dir,
                            "mulran": config.mulran_dir,
                            "ugv" : config.ugv_dir,
                            "apollo" : config.apollo_dir,
                            "bushwalk" : config.bushwalk_dir}

        self.eval_pickle = config.eval_pickle

        PointCloudDataset.__init__(
            self, phase, random_rotation, random_occlusion, random_scale, config)

        logging.info("Initializing GeneralDataset")
        # for pickle_dataset, pickle_path in self.eval_pickle.items():
        pickle_dataset = 'bushwalk'
        logging.info(f"Loading the pickles for {pickle_dataset} from {self.eval_pickle}")
        tuple_set = pickle.load(open(self.eval_pickle, 'rb'))
        for train_tuple in tuple_set:
            tuple = GeneralTrainingTuple(dataset = pickle_dataset, training_tuple=tuple_set[train_tuple])
            self.files.append(tuple)


    def get_pointcloud_tensor(self, base_dir, rel_path, dataset):
        fname = os.path.join(base_dir, rel_path)
        pcd = o3d.io.read_point_cloud(fname) # TODO: add numpy load, conditional
        downpcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        xyz = np.asarray(downpcd.points)
        oo = np.ones(len(xyz)).reshape((-1,1))
        xyzr = np.hstack((xyz, oo)).astype(np.float32)

        return xyzr

    def __getitem__(self, idx):
        anchor_data = self.files[idx]
        
        xyz0_th = self.get_pointcloud_tensor(self.base_dirs[anchor_data.dataset], anchor_data.rel_scan_filepath, anchor_data.dataset)
        meta_info = {'drive': anchor_data.dataset, 't': anchor_data.timestamp}

        return (xyz0_th,
                meta_info)
class GeneralDataset(PointCloudDataset):
    r"""
    Generate single pointcloud frame from General dataset. 
    """

    def __init__(self,
                 phase,
                 random_rotation=False,
                 random_occlusion=False,
                 random_scale=False,
                 config=None):

        # self.root = root = config.general_dir
        self.base_dirs = {"kitti" : config.kitti_dir,
                            "mulran": config.mulran_dir,
                            "ugv" : config.ugv_dir,
                            "apollo" : config.apollo_dir,
                            "bushwalk" : config.bushwalk_dir}

        self.train_pickles = config.train_pickles
        self.pnv_prep = config.pnv_preprocessing
        self.gp_rem = config.gp_rem
        self.gp_vals = config.gp_vals
        self.voxel_size = config.voxel_size

        PointCloudDataset.__init__(
            self, phase, random_rotation, random_occlusion, random_scale, config)

        logging.info("Initializing GeneralDataset")
        for pickle_dataset, pickle_path in self.train_pickles.items():
            logging.info(f"Loading the pickles for {pickle_dataset} from {pickle_path}")
            tuple_set = pickle.load(open(pickle_path, 'rb'))
            for train_tuple in tuple_set:
                tuple = GeneralTrainingTuple(dataset = 'apollo', training_tuple=tuple_set[train_tuple])
                self.files.append(tuple)


    def get_pointcloud_tensor(self, base_dir, rel_path, dataset):
        fname = os.path.join(base_dir, rel_path)
        pcd = o3d.io.read_point_cloud(fname) # TODO: add numpy load, conditional
        downpcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        xyz = np.asarray(downpcd.points)
        oo = np.ones(len(xyz)).reshape((-1,1))
        xyzr = np.hstack((xyz, oo)).astype(np.float32)

        if self.gp_rem:
            gp_val = self.gp_vals[dataset]
            xyzr = xyzr[xyzr[:,2] > - gp_val]

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
        anchor_data = self.files[idx]
        
        xyz0_th = self.get_pointcloud_tensor(self.base_dirs[anchor_data.dataset], anchor_data.rel_scan_filepath, anchor_data.dataset)
        meta_info = {'drive': anchor_data.dataset, 't': anchor_data.timestamp}

        return (xyz0_th,
                meta_info)


class GeneralTupleDataset(GeneralDataset):
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
        self.base_dirs = {"kitti" : config.kitti_dir,
                            "mulran": config.mulran_dir,
                            "ugv" : config.ugv_dir,
                            "apollo" : config.apollo_dir,
                            "bushwalk" : config.bushwalk_dir}

        self.train_pickles = config.train_pickles
        self.positives_per_query = config.positives_per_query
        self.negatives_per_query = config.negatives_per_query
        self.quadruplet = False
        self.pnv_prep = config.pnv_preprocessing
        self.gp_rem = config.gp_rem
        self.gp_vals = config.gp_vals
        self.voxel_size = config.voxel_size
        self.offsets = []
        if config.train_loss_function == 'quadruplet':
            self.quadruplet = True

        PointCloudDataset.__init__(
            self, phase, random_rotation, random_occlusion, random_scale, config)

        logging.info("Initializing GeneralTupleDataset")
        id_offset=0
        self.offsets.append(id_offset)
        for pickle_dataset, pickle_path in self.train_pickles.items():
            logging.info(f"Loading the pickles for {pickle_dataset} from {pickle_path}")
            tuple_set = pickle.load(open(pickle_path, 'rb'))
            for tuple_id in range(len(tuple_set)):
                train_tuple = tuple_set[tuple_id]
                tuple = GeneralTrainingTuple(dataset = pickle_dataset, training_tuple=train_tuple, id_offset=id_offset)
                self.files.append(tuple)
            id_offset += len(tuple_set)
            self.offsets.append(id_offset)
        print('')
        

    def get_positives(self, idx):
        return self.files[idx].positives

    def get_negatives(self, query_idx, non_negative_ids):
        offset_id = next(x[0] for x in enumerate(self.offsets) if x[1] > query_idx) - 1
        all_ids = set(np.arange(self.offsets[offset_id], self.offsets[offset_id+1]))
        neg_set = all_ids.difference(list(non_negative_ids))
        negatives = list(neg_set)
        return negatives

    def get_other_negative(self, drive_id,query_id, sel_positive_ids, sel_negative_ids):
        # Dissimillar to all pointclouds in triplet tuple.
        offset_id = next(x[0] for x in enumerate(self.offsets) if x[1] > query_id) - 1
        all_ids = set(np.arange(self.offsets[offset_id], self.offsets[offset_id+1]))
        neighbour_ids = sel_positive_ids
        for neg_idx in sel_negative_ids:
            neg_postives_files = self.get_positives(neg_idx)
            for pos in neg_postives_files:
                neighbour_ids.append(pos)
        possible_negs = list(all_ids - set(neighbour_ids))
        if query_id in possible_negs:
            possible_negs.remove(query_id)
        assert len(
            possible_negs) > 0, f"No other negatives for drive {drive_id} id {query_id}"
        other_neg_id = random.sample(possible_negs, 1)
        return other_neg_id[0]

    def __getitem__(self, idx):
        anchor_data = self.files[idx]
        positive_ids, non_negative_ids = anchor_data.positives, anchor_data.non_negatives
        negative_ids = self.get_negatives(idx, non_negative_ids)

        sel_positive_ids = random.sample(
            list(positive_ids), self.positives_per_query)
        sel_negative_ids = random.sample(
            list(negative_ids), self.negatives_per_query)
        positives, negatives, other_neg = [], [], None

        query_th = self.get_pointcloud_tensor(self.base_dirs[anchor_data.dataset], anchor_data.rel_scan_filepath, anchor_data.dataset)
        for sp_id in sel_positive_ids:
            sp_data = self.files[sp_id]
            positives.append(self.get_pointcloud_tensor(self.base_dirs[anchor_data.dataset], sp_data.rel_scan_filepath, anchor_data.dataset))
        for sn_id in sel_negative_ids:
            sn_data = self.files[sn_id]
            negatives.append(self.get_pointcloud_tensor(self.base_dirs[anchor_data.dataset], sn_data.rel_scan_filepath, anchor_data.dataset))

        meta_info = {'drive': anchor_data.dataset, 'query_id': anchor_data.id}

        if not self.quadruplet:
            return (query_th,
                    positives,
                    negatives,
                    meta_info)
        else:  # For Quadruplet Loss
            other_neg_id = self.get_other_negative(anchor_data.dataset, idx, sel_positive_ids, sel_negative_ids)
            other_neg_data = self.files[other_neg_id]
            other_neg_th = self.get_pointcloud_tensor(self.base_dirs[anchor_data.dataset], other_neg_data.rel_scan_filepath, anchor_data.dataset)
            return (query_th,
                    positives,
                    negatives,
                    other_neg_th,
                    meta_info)

