import os
import sys
import torch
import torch.nn as nn
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from models.aggregators.NetVLAD import *
from models.backbones.PointNet import *

class PointNetVLAD(nn.Module):
    def __init__(self, num_points=2500, global_feat=True, feature_transform=False, max_pool=True, output_dim=1024):
        super(PointNetVLAD, self).__init__()
        self.point_net = PointNetfeat(num_points=num_points, global_feat=global_feat,
                                      feature_transform=feature_transform, max_pool=max_pool)
        self.net_vlad = NetVLADLoupe(feature_size=1024, max_samples=num_points, cluster_size=64,
                                     output_dim=output_dim, gating=True, add_batch_norm=True,
                                     is_training=True)

    def forward(self, x):
        x = self.point_net(x)
        x = self.net_vlad(x)
        return x


if __name__ == '__main__':
    num_points = 4096
    sim_data = Variable(torch.rand(44, 1, num_points, 3))

    pnv = PointNetVLAD(global_feat=True, feature_transform=True,
                       max_pool=False, output_dim=256, num_points=num_points)  # .cuda()
    pnv.train()
    out = pnv(sim_data)
    print('pnv', out.size())
