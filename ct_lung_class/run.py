import os 
import torch

import torch.distributed as dist
import torch.multiprocessing as mp

def create_argument_parser():
    pass

class Trainer:
    pass

def ddp_setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def ddp_cleanup():
    dist.destroy_process_group()
    
    
def main(rank: int, world_size: int, cli_args: dict):
    ddp_setup(rank, world_size)
    trainer = Trainer(rank, world_size, cli_args)
    trainer.train()
    ddp_cleanup


if __name__ == "__main__":
    parser = create_argument_parser()
    cli_args = parser.parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, cli_args), nprocs=world_size)