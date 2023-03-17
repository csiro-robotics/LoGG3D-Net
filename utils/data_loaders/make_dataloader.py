import logging
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from torch.utils.data.sampler import Sampler
from utils.data_loaders.mulran.mulran_sparse_dataset import *
from utils.data_loaders.mulran.mulran_dataset import *
from utils.data_loaders.kitti.kitti_sparse_dataset import *
from utils.data_loaders.kitti.kitti_dataset import *


ALL_DATASETS = [
    KittiDataset, KittiTupleDataset, KittiSparseTupleDataset, KittiPointSparseTupleDataset,
    MulRanDataset, MulRanTupleDataset, MulRanSparseTupleDataset, MulRanPointSparseTupleDataset
]
dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}


class RandomSampler(Sampler):
    """Samples elements randomly, without replacement.
      Arguments:
          data_source (Dataset): dataset to sample from
          shuffle: use random permutation
      """

    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        else:
            perm = torch.arange(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop(0)

    def __len__(self):
        return len(self.data_source)


def make_data_loader(config, phase, batch_size, num_workers=0, shuffle=None, dist=None):
    assert phase in ['train', 'trainval', 'val', 'test']
    if shuffle is None:
        shuffle = phase != 'test'

    use_random_scale = False
    use_random_rotation = False
    use_random_occlusion = False
    if phase in ['train', 'trainval']:
        use_random_rotation = config.use_random_rotation
        use_random_scale = config.use_random_scale
        use_random_occlusion = config.use_random_occlusion
        Dataset = dataset_str_mapping[config.dataset]

    elif phase in ['val', 'test']:
        use_random_rotation = config.eval_random_rotation
        use_random_occlusion = config.eval_random_occlusion
        Dataset = dataset_str_mapping[config.eval_dataset]

    dset = Dataset(phase,
                   random_scale=use_random_scale,
                   random_rotation=use_random_rotation,
                   random_occlusion=use_random_occlusion,
                   config=config)

    collation_type = config.collation_type
    if (phase in ['train', 'trainval']):
        if ('Tuple' in config.dataset):
            collation_type = 'tuple'
        if ('SparseTuple' in config.dataset):
            collation_type = 'sparse_tuple'
        if ('PointSparseTuple' in config.dataset):
            collation_type = 'reg_sparse_tuple'
    collation_fn = CollationFunctionFactory(
        collation_type, config.voxel_size, config.num_points)

    if dist:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dset,
            num_replicas=dist[0],
            rank=dist[1],
            shuffle=(phase == 'train'))
    else:
        sampler = RandomSampler(dset, shuffle)
        logging.info('collation_type: ' + str(collation_type))
        logging.info('num_workers: ' + str(num_workers))
        logging.info('shuffle: ' + str(shuffle))
        logging.info('use_random_rotation: ' + str(use_random_rotation))
        logging.info('use_random_occlusion: ' + str(use_random_occlusion))
        logging.info('use_random_scale: ' + str(use_random_scale))

    loader = torch.utils.data.DataLoader(dset,
                                         batch_size=batch_size,
                                         collate_fn=collation_fn,
                                         num_workers=num_workers,
                                         sampler=sampler,
                                         pin_memory=True)

    return loader


if __name__ == "__main__":
    from config.eval_config import get_config_eval
    from utils.o3d_tools import *

    logger = logging.getLogger()
    cfg = get_config_eval()

    test_loader = make_data_loader(cfg,
                                   cfg.test_phase,
                                   cfg.eval_batch_size,
                                   num_workers=cfg.test_num_workers,
                                   shuffle=True)

    iterator = test_loader.__iter__()
    # s = iterator.size()
    l = len(test_loader.dataset)
    for i in range(l):
        input_data = next(iterator)
        quad = True
        if quad:
            visualize_scan_open3d(input_data[0][1][0][:, :3])
            visualize_scan_open3d(input_data[0][2][0][:, :3])
            visualize_scan_open3d(input_data[0][3][:, :3])
        else:
            visualize_scan_open3d(input_data[0][0][:, :3])
    print('')
