import os
import sys
import torch
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchpack import distributed as dist
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.misc_utils import log_config
from evaluation.evaluate import *
from utils.data_loaders.make_dataloader import *
from config.train_config import get_config
from models.pipeline_factory import get_pipeline
from training import train_utils

# from models.backbones.spvnas.core.modules import dist
cfg = get_config()

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    handlers=[ch])
logging.basicConfig(level=logging.INFO, format="")


def main():
    dist.init()
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    if (dist.rank() % dist.size() == 0):
        writer = SummaryWriter(comment=f"_{cfg.job_id}")

        logger = logging.getLogger()
        logging.info('\n' + ' '.join([sys.executable] + sys.argv))
        logging.info('Slurm Job ID: ' + cfg.job_id)
        logging.info('Training pipeline: ' + cfg.train_pipeline)
        log_config(cfg, logging)
        cfg.experiment_name = f"{datetime.now(tz=None).strftime('%Y-%m-%d_%H-%M-%S')}_{cfg.experiment_name}_{cfg.job_id}"
        logging.info("Experiment Name: " + cfg.experiment_name)

        logging.info('dist size: ' + str(dist.size()))
        logging.info('dist rank: ' + str(dist.rank()))

    # Get model
    model = get_pipeline(cfg.train_pipeline)
    n_params = sum([param.nelement() for param in model.parameters()])
    logging.info('Number of model parameters: {}'.format(n_params))

    # Get train utils
    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    loss_function = train_utils.get_loss_function(cfg)
    point_loss_function = train_utils.get_point_loss_function(cfg)
    optimizer = train_utils.get_optimizer(cfg, model.parameters())
    scheduler = train_utils.get_scheduler(cfg, optimizer)

    if cfg.resume_training:
        resume_filename = cfg.resume_checkpoint
        save_path = os.path.join(os.path.dirname(
            __file__), 'checkpoints', resume_filename)
        print("Resuming Model From ", save_path)
        checkpoint = torch.load(save_path)
        starting_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        starting_epoch = 0

    model = torch.nn.parallel.DistributedDataParallel(
        model.to('cuda:%d' % dist.local_rank()),
        device_ids=[dist.local_rank()],
        find_unused_parameters=True)

    # Get data loader
    train_loader = make_data_loader(cfg,
                                    cfg.train_phase,
                                    cfg.batch_size,
                                    num_workers=cfg.train_num_workers,
                                    shuffle=True,
                                    dist=[dist.size(), dist.rank()])

    for epoch in range(starting_epoch, cfg.max_epoch):
        if (dist.rank() % dist.size() == 0):
            lr = scheduler.get_last_lr()
            logging.info('\n' + '**** EPOCH %03d ****' %
                         (epoch) + ' LR: %03f' % (lr[0]))
        running_loss = 0.0
        running_scene_loss = 0.0
        running_point_loss = 0.0
        model.train()

        for i, batch in enumerate(train_loader, 0):
            if cfg.train_pipeline == 'LOGG3D':
                batch_st = batch[0].to('cuda:%d' % dist.local_rank())
                if not batch[1]['pos_pairs'].ndim == 2:
                    continue
                output = model(batch_st)
                scene_loss = loss_function(output[0], cfg)
                running_scene_loss += scene_loss.item()
                if cfg.point_loss_weight > 0:
                    point_loss = point_loss_function(
                        output[1][0], output[1][1], batch[1]['pos_pairs'], cfg)
                    running_point_loss += point_loss.item()
                    loss = cfg.scene_loss_weight * scene_loss + cfg.point_loss_weight * point_loss
                else:
                    loss = scene_loss

            elif cfg.train_pipeline == 'PointNetVLAD':
                batch_t = batch.to('cuda:%d' % dist.local_rank())
                output = model(batch_t.unsqueeze(1))
                scene_loss = loss_function(output, cfg)
                running_scene_loss += scene_loss.item()
                loss = scene_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i % cfg.loss_log_step) == (cfg.loss_log_step - 1):
                avg_loss = running_loss / cfg.loss_log_step
                avg_scene_loss = running_scene_loss / cfg.loss_log_step
                avg_point_loss = running_point_loss / cfg.loss_log_step

                if (dist.rank() % dist.size() == 0):
                    lr = scheduler.get_last_lr()
                    logging.info('avg running loss: ' +
                                 str(avg_loss) + ' LR: %03f' % (lr[0]))
                    logging.info('avg_scene_loss: ' + str(avg_scene_loss) +
                                 ' avg_point_loss: ' + str(avg_point_loss))
                    writer.add_scalar('training loss',
                                      avg_loss,
                                      epoch * len(train_loader) + i)
                    writer.add_scalar('training point loss',
                                      avg_point_loss,
                                      epoch * len(train_loader) + i)
                    writer.add_scalar('training scene loss',
                                      avg_scene_loss,
                                      epoch * len(train_loader) + i)
                running_loss, running_scene_loss, running_point_loss = 0.0, 0.0, 0.0

        scheduler.step()

        if cfg.save_model_after_epoch and (dist.rank() % dist.size() == 0):
            save_path = os.path.join(os.path.dirname(__file__), 'checkpoints')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = str(save_path) + '/' + cfg.experiment_name
            logging.info("Saving to: " + str(save_path))
            if isinstance(model, torch.nn.DataParallel):
                model_to_save = model.module
            elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model_to_save = model.module
            else:
                model_to_save = model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            },
                save_path)

    logging.info("Finished training.")


if __name__ == "__main__":
    main()
