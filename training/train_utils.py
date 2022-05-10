import os
import sys
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from loss.global_loss import *
from loss.local_consistency_loss import *

def get_loss_function(cfg):
    if cfg.train_loss_function == 'triplet':
        loss_function = triplet_loss
    elif cfg.train_loss_function == 'quadruplet':
        loss_function = quadruplet_loss
    else:
        raise NotImplementedError(cfg.train_loss_function)
    return loss_function

def get_point_loss_function(cfg):
    if cfg.point_loss_function == 'contrastive':
        point_loss_function = point_contrastive_loss
    elif cfg.point_loss_function == 'infonce':
        point_loss_function = point_infonce_loss
    else:
        raise NotImplementedError(cfg.point_loss_function)
    return point_loss_function   

def get_optimizer(cfg, model_parameters):
    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model_parameters, cfg.base_learning_rate, momentum=cfg.momentum)
    elif cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(model_parameters, cfg.base_learning_rate)  
    else:
        raise NotImplementedError(cfg.optimizer)
    return optimizer 

def get_scheduler(cfg, optimizer):
    # See: https://www.kaggle.com/isbhargav/guide-to-pytorch-learning-rate-scheduling
    if cfg.scheduler == 'lambda':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1)
    elif cfg.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10)
    elif cfg.scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30,40], gamma=0.1)
    else:
        raise NotImplementedError(cfg.scheduler)
    return scheduler
