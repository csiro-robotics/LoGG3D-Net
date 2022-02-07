import os
import sys
import torch
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from evaluation.eval_sequence import *
from utils.misc_utils import log_config

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    handlers=[ch])
logging.basicConfig(level=logging.INFO, format="")

def evaluate_checkpoint(model, save_path, cfg):
  checkpoint = torch.load(save_path)#,map_location='cuda:0')
  model.load_state_dict(checkpoint['model_state_dict'])

  epoch = checkpoint['epoch']
  loss = checkpoint['loss']
  
  model = model.cuda()
  model.eval()

  return evaluate_sequence_reg(model, cfg) 


if __name__ == "__main__":
  from models.pipeline_factory import get_pipeline
  from config.eval_config import get_config_eval

  cfg = get_config_eval()

  # Get model
  model = get_pipeline(cfg.eval_pipeline)

  save_path = os.path.join(os.path.dirname(__file__), '../', 'checkpoints')
  save_path = str(save_path) + cfg.checkpoint_name 
  print('Loading checkpoint from: ', save_path)
  logging.info('\n' + ' '.join([sys.executable] + sys.argv))
  log_config(cfg, logging)

  eval_F1_max = evaluate_checkpoint(model, save_path, cfg)
  logging.info('\n' + '******************* Evaluation Complete *******************')
  logging.info('Checkpoint Name: ' + str(cfg.checkpoint_name))
  if 'Kitti' in cfg.eval_dataset:
    logging.info('Evaluated Sequence: ' + str(cfg.kitti_eval_seq))
  elif 'MulRan'  in cfg.eval_dataset:
    logging.info('Evaluated Sequence: ' + str(cfg.mulran_eval_seq))
  logging.info('F1 Max: ' + str(eval_F1_max))
