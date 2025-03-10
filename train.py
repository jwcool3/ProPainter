import os
import json
import argparse
import subprocess

from shutil import copyfile
import torch.distributed as dist

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import core
import core.trainer
import core.trainer_flow_w_edge

from core.dist import (
    get_world_size,
    get_local_rank,
    get_global_rank,
    get_master_ip,
)

parser = argparse.ArgumentParser()
parser.add_argument('-c',
                    '--config',
                    default='configs/train_propainter.json',
                    type=str)
parser.add_argument('-p', '--port', default='23490', type=str)
args = parser.parse_args()


def main_worker(rank, world_size, config):
    config['local_rank'] = config['global_rank'] = rank
    torch.cuda.set_device(rank)
    config['device'] = torch.device(f"cuda:{rank}")

    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    config['save_dir'] = os.path.join(
        config['save_dir'],
        '{}_{}'.format(config['model']['net'],
                       os.path.basename(args.config).split('.')[0]))

    config['save_metric_dir'] = os.path.join(
        './scores',
        '{}_{}'.format(config['model']['net'],
                       os.path.basename(args.config).split('.')[0]))

    if rank == 0:
        os.makedirs(config['save_dir'], exist_ok=True)
        config_path = os.path.join(config['save_dir'],
                                   os.path.basename(args.config))
        if not os.path.isfile(config_path):
            copyfile(args.config, config_path)
        print('[**] create folder {}'.format(config['save_dir']))

    trainer_version = config['trainer']['version']
    trainer = core.__dict__[trainer_version].__dict__['Trainer'](config)
    trainer.train()

    dist.destroy_process_group()


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    mp.set_sharing_strategy('file_system')

    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # loading configs
    config = json.load(open(args.config))

    config['world_size'] = 2
    config['distributed'] = True
    print('world_size:', config['world_size'])

    mp.spawn(main_worker, args=(config['world_size'], config), nprocs=config['world_size'], join=True)
