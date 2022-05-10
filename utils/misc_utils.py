""" Miscellaneous functions """
import numpy as np
from numpy import dot
from numpy.linalg import norm
import time
import torch

#####################################################################################
# Logging


def log_config(cfg, logging):
    for gpu_id in range(torch.cuda.device_count()):
        logging.info(str(gpu_id) + ' ' + torch.cuda.get_device_name(gpu_id))
    logging.info('\n' + '===> Configurations')
    dconfig = vars(cfg)
    for k in dconfig:
        logging.info('    {}: {}'.format(k, dconfig[k]))
    logging.info('\n' + '\n')

#####################################################################################
# Place recognition


def check_if_revisit(query_pose, db_poses, thres, return_id=False):
    num_dbs = np.shape(db_poses)[0]
    is_revisit = 0

    for i in range(num_dbs):
        dist = norm(query_pose - db_poses[i])
        if (dist < thres):
            is_revisit = 1
            break

    if return_id:
        return is_revisit, i
    else:
        return is_revisit

#####################################################################################
# Math


def cosine_distance(feature_a, feature_b):
    return 1 - dot(feature_a, np.transpose(feature_b))/(norm(feature_a)*norm(feature_b))


def T_inv(T_in):
    """ Return the inverse of input homogeneous transformation matrix """
    R_in = T_in[:3, :3]
    t_in = T_in[:3, [-1]]
    R_out = R_in.T
    t_out = -np.matmul(R_out, t_in)
    return np.vstack((np.hstack((R_out, t_out)), np.array([0, 0, 0, 1])))


def is_nan(x):
    return (x != x)


def euclidean_to_homogeneous(e_point):
    """ Coversion from Eclidean coordinates to Homogeneous """
    h_point = np.concatenate([e_point, [1]])
    return h_point


def homogeneous_to_euclidean(h_point):
    """ Coversion from Homogeneous coordinates to Eclidean """
    e_point = h_point / h_point[3]
    e_point = e_point[:3]
    return e_point


def hashM(arr, M):
    if isinstance(arr, np.ndarray):
        N, D = arr.shape
    else:
        N, D = len(arr[0]), len(arr)

    hash_vec = np.zeros(N, dtype=np.int64)
    for d in range(D):
        if isinstance(arr, np.ndarray):
            hash_vec += arr[:, d] * M**d
        else:
            hash_vec += arr[d] * M**d
    return hash_vec


def pdist(A, B, dist_type='L2'):
    if dist_type == 'L2':
        D2 = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
        return torch.sqrt(D2 + 1e-7)
    elif dist_type == 'SquareL2':
        return torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
    else:
        raise NotImplementedError('Not implemented')

#####################################################################################
# Timing


class Timer(object):
    """A simple timer."""
    # Ref: https://github.com/chrischoy/FCGF/blob/master/lib/timer.py

    def __init__(self, binary_fn=None, init_val=0):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.binary_fn = binary_fn
        self.tmp = init_val

    def reset(self):
        self.total_time = 0
        self.calls = 0
        self.start_time = 0
        self.diff = 0

    @property
    def avg(self):
        return self.total_time / self.calls

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        if self.binary_fn:
            self.tmp = self.binary_fn(self.tmp, self.diff)
        if average:
            return self.avg
        else:
            return self.diff

#####################################################################################
# Config


font = {'family': 'serif',
        # 'color':  'black',
        'weight': 'normal',
        'size': 16,
        }
font_legend = {'family': 'serif',
               # 'color':  'black',
               'weight': 'normal',
               'size': 12,
               }
