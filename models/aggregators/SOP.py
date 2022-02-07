from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SOP(nn.Module):
    def __init__(self, thresh=1e-8, is_vec = True, signed_sqrt = False, do_pe=True, do_fc=False, input_dim=16, is_tuple=False):
        super(SOP, self).__init__()
        self.thresh = thresh
        self.is_vec = is_vec
        self.do_pe = do_pe
        self.sop_dim = input_dim * input_dim
        self.signed_sqrt = signed_sqrt
        self.do_fc = do_fc
        self.is_tuple = is_tuple
        
        cs = [4096, 2048, 1024] # redundant fc layers
        cr = self.sop_dim/cs[0]
        cs = [int(cr * x) for x in cs]
        self.fc1 = nn.Linear(cs[0], cs[1])
        self.fc2 = nn.Linear(cs[1], cs[2])


    def _so_maxpool(self, x):
        while len(x.data.shape ) < 4:
            x = torch.unsqueeze(x, 0)
        # x = x.double()
        batchSize, tupleLen, nFeat, dimFeat = x.data.shape 
        x = torch.reshape(x, (-1, dimFeat))
        x = torch.unsqueeze(x, -1)
        x = x.matmul(x.transpose(1, 2))

        x = torch.reshape(x, (batchSize, tupleLen, nFeat, dimFeat, dimFeat))
        x = torch.max(x, 2).values
        x = torch.reshape(x, (-1, dimFeat, dimFeat))
        if self.do_pe:
            # u_, s_, vh_ = torch.linalg.svd(x)
            # dist = torch.dist(x, u_ @ torch.diag_embed(s_) @ vh_)
            # dist_same = torch.allclose(x, u_ @ torch.diag_embed(s_) @ vh_)
            # s_alpha = torch.pow(s_, 0.5)
            # x = u_ @ torch.diag_embed(s_alpha) @ vh_
            x = x.double()
            u_, s_, v_ = torch.svd(x) # For pytorch versions < 1.9
            # dist = torch.dist(x, u_ @ torch.diag_embed(s_) @  v_.transpose(-2, -1))
            # dist_same = torch.allclose(x, u_ @ torch.diag_embed(s_) @  v_.transpose(-2, -1))
            s_alpha = torch.pow(s_, 0.5)
            x = u_ @ torch.diag_embed(s_alpha) @ v_.transpose(-2, -1)

        x = torch.reshape(x, (batchSize, tupleLen, dimFeat, dimFeat))
        return x#.float()

    def _l2norm(self, x):
        x = nn.functional.normalize(x, p=2, dim=-1)
        return x

    def forward(self, x):
        x = self._so_maxpool(x)
        
        if self.is_vec:
            x = torch.reshape(x, (x.size(0),x.size(1),-1))
        # if self.do_fc:
        #     x = F.relu(self.fc1(x.float()))
        #     x = F.relu(self.fc2(x))
        x = self._l2norm(x)
        return torch.squeeze(x)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = SOP(signed_sqrt = False, do_fc=False)
    model = model.to(device)
    model = nn.DataParallel(model)
    segments = np.random.rand(44,100,64)*10
    feed_tensor = torch.from_numpy(segments).float()#s.double()
    feed_tensor = torch.unsqueeze(feed_tensor, 0)
    feed_tensor = torch.reshape(feed_tensor, (2, -1, 100, 64))
    feed_tensor = feed_tensor.to(device)
    output = model(feed_tensor)
    print('')