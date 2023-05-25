import os
import sys
sys.path.append(os.path.dirname(__file__))
from pipelines.PointNetVLAD import *
from pipelines.LOGG3D import *


def get_pipeline(pipeline_name):
    if pipeline_name == 'LOGG3D':
        pipeline = LOGG3D(feature_dim=16)
    elif pipeline_name == 'PointNetVLAD':
        pipeline = PointNetVLAD(global_feat=True, feature_transform=True,
                                max_pool=False, output_dim=256, num_points=4096)
    return pipeline


if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
    from config.eval_config import get_config
    cfg = get_config()
    model = get_pipeline(cfg.train_pipeline).cuda()
    # print(model)

    from utils.data_loaders.make_dataloader import *
    train_loader = make_data_loader(cfg,
                                    cfg.train_phase,
                                    cfg.batch_size,
                                    num_workers=cfg.train_num_workers,
                                    shuffle=True)
    iterator = train_loader.__iter__()
    l = len(train_loader.dataset)
    for i in range(l):
        input_batch = next(iterator)
        input_st = input_batch[0].cuda()
        output = model(input_st)
        print('')
