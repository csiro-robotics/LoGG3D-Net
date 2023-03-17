import sys
import os
import torch
import numpy as np
from torchsparse.utils.collate import sparse_collate
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.o3d_tools import *

class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self,
                 phase,
                 random_rotation=False,
                 random_occlusion=False,
                 random_scale=False,
                 config=None):
        self.phase = phase
        self.files = []

        self.random_scale = random_scale
        self.min_scale = config.min_scale
        self.max_scale = config.max_scale
        self.random_occlusion = random_occlusion
        self.random_rotation = random_rotation
        self.rotation_range = config.rotation_range
        if random_rotation:
            print('***********************Dataloader initialized with Random Rotation. ')
        if random_occlusion:
            print('***********************Dataloader initialized with Random Occlusion. ')
        if random_scale:
            print('***********************Dataloader initialized with Random Scale. ')

    def random_rotate(self, xyzr, r_angle=360, is_random=True, add_noise=True, rand_tr=False):
        # If is_random = True: Rotate about z-axis by random angle upto 'r_angle'.
        # Else: Rotate about z-axis by fixed angle 'r_angle'.
        r_angle = (np.pi/180) * r_angle
        if is_random:
            r_angle = r_angle*np.random.uniform()
        cos_angle = np.cos(r_angle)
        sin_angle = np.sin(r_angle)
        rot_matrix = np.array([[cos_angle, -sin_angle, 0],
                               [sin_angle, cos_angle, 0],
                               [0,             0,      1]])
        scan = xyzr[:, :3]
        int = xyzr[:, 3].reshape((-1, 1))
        augmented_scan = np.dot(scan, rot_matrix)

        if add_noise:
            n_sigma = 0.01  # Add gaussian noise
            noise = np.clip(n_sigma * np.random.randn(*
                            augmented_scan.shape), -0.03, 0.03)
            augmented_scan = augmented_scan + noise

        if rand_tr:
            tr_xy_max, tr_z_max = 1.5, 0.25
            tr_xy = np.clip(np.random.randn(1, 2), -tr_xy_max, tr_xy_max)
            tr_z = np.clip(0.1*np.random.randn(1, 1), -tr_z_max, tr_z_max)
            tr = np.hstack((tr_xy, tr_z))
            augmented_scan = augmented_scan + tr

        augmented_scan = np.hstack((augmented_scan, int))
        return augmented_scan.astype(np.float32)

    def occlude_scan(self, scan, angle=30):
        # Remove points within a sector of fixed angle (degrees) and random heading direction.
        thetas = (180/np.pi) * np.arctan2(scan[:, 1], scan[:, 0])
        heading = (180-angle/2)*np.random.uniform(-1, 1)
        occ_scan = np.vstack(
            (scan[thetas < (heading - angle/2)], scan[thetas > (heading + angle/2)]))
        return occ_scan.astype(np.float32)

    def pnv_preprocessing(self, xyzr, l=25):
        ind = np.argwhere(xyzr[:, 0] <= l).reshape(-1)
        xyzr = xyzr[ind]
        ind = np.argwhere(xyzr[:, 0] >= -l).reshape(-1)
        xyzr = xyzr[ind]
        ind = np.argwhere(xyzr[:, 1] <= l).reshape(-1)
        xyzr = xyzr[ind]
        ind = np.argwhere(xyzr[:, 1] >= -l).reshape(-1)
        xyzr = xyzr[ind]
        ind = np.argwhere(xyzr[:, 2] <= l).reshape(-1)
        xyzr = xyzr[ind]
        ind = np.argwhere(xyzr[:, 2] >= -l).reshape(-1)
        xyzr = xyzr[ind]

        vox_sz = 0.3
        while len(xyzr) > 4096:
            xyzr = downsample_point_cloud(xyzr, vox_sz)
            vox_sz += 0.01

        if xyzr.shape[0] >= 4096:
            ind = np.random.choice(xyzr.shape[0], 4096, replace=False)
            xyzr = xyzr[ind, :]
        else:
            ind = np.random.choice(xyzr.shape[0], 4096, replace=True)
            xyzr = xyzr[ind, :]
        mean = np.mean(xyzr, axis=0)
        pc = xyzr - mean
        scale = np.max(abs(pc))
        pc = pc/scale
        return pc[:,:3]

    def __len__(self):
        return len(self.files)


class CollationFunctionFactory:
    def __init__(self, collation_type='default', voxel_size=0.05, num_points=80000):
        self.voxel_size = voxel_size
        self.num_points = num_points
        if collation_type == 'default':
            self.collation_fn = self.collate_default
        elif collation_type == 'tuple':
            self.collation_fn = self.collate_tuple
        elif collation_type == 'sparse_tuple':
            self.collation_fn = self.collate_sparse_tuple
        elif collation_type == 'reg_sparse_tuple':
            self.collation_fn = self.collate_reg_sparse_tuple
        elif collation_type == 'sparcify_list':
            self.collation_fn = self.sparcify_and_collate_list
        else:
            raise ValueError(f'collation_type {collation_type} not found')

    def __call__(self, list_data):
        return self.collation_fn(list_data)

    def collate_default(self, list_data):
        if len(list_data) > 1:
            return self.collate_tuple(list_data)
        else:
            return list_data

    def collate_tuple(self, list_data):
        outputs = []
        for batch_data in list_data:
            contrastive_tuple = []
            for tuple_data in batch_data:
                if isinstance(tuple_data, np.ndarray):
                    contrastive_tuple.append(tuple_data)
                elif isinstance(tuple_data, list):
                    contrastive_tuple.extend(tuple_data)
            # outputs.append(sparse_collate(contrastive_tuple))
        outputs = [torch.from_numpy(ct).float() for ct in contrastive_tuple]
        outputs = torch.stack(outputs)
        return outputs

    def collate_sparse_tuple(self, list_data):
        outputs = []
        for tuple_data in list_data:
            contrastive_tuple = []
            for name in tuple_data.keys():
                if isinstance(tuple_data[name], SparseTensor):
                    contrastive_tuple.append(tuple_data[name])
                elif isinstance(tuple_data[name], (list, np.ndarray)):
                    contrastive_tuple.extend(tuple_data[name])
            outputs.append(sparse_collate(contrastive_tuple))
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    def collate_reg_sparse_tuple(self, list_data):
        outputs = []
        for tuple_data in list_data:
            contrastive_tuple = []
            meta_info = None
            for name in tuple_data.keys():
                if isinstance(tuple_data[name], SparseTensor):
                    contrastive_tuple.append(tuple_data[name])
                elif isinstance(tuple_data[name], (list, np.ndarray)):
                    contrastive_tuple.extend(tuple_data[name])
                elif isinstance(tuple_data[name], dict):
                    meta_info = tuple_data[name]
            outputs.append([sparse_collate(contrastive_tuple), meta_info])
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    def sparcify_and_collate_list(self, list_data):
        outputs = []
        if isinstance(list_data, SparseTensor):
            return list_data
        else:
            # return outputs
            for xyzr in list_data:
                xyzr = xyzr[0]
                if not len(xyzr) > 0:
                    continue
                pc_ = np.round(xyzr[:, :3] / self.voxel_size).astype(np.int32)
                pc_ -= pc_.min(0, keepdims=1)
                feat_ = xyzr

                _, inds, inverse_map = sparse_quantize(pc_,
                                                       return_index=True,
                                                       return_inverse=True)
                if len(inds) > self.num_points:
                    inds = np.random.choice(
                        inds, self.num_points, replace=False)

                pc = pc_[inds]
                feat = feat_[inds]
                outputs.append(SparseTensor(feat, pc))
            return sparse_collate(outputs)
