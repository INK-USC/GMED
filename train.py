import argparse
import os
import sys

import torch
import numpy as np
import random
from trainer_benchmark import ocl_train_mnist, ocl_train_cifar
from utils.utils import get_exp_id, set_config_attr, get_config_attr
from yacs.config import CfgNode

from nets.classifier import ResNetClassifier, ResNetClassifierWObj
from nets.simplenet import mnist_simple_net_400width_classlearning_1024input_10cls_1ds


from ocl import NaiveWrapper, ExperienceReplay, AGEM, ExperienceEvolveApprox


def train(cfg, local_rank, distributed, tune=False):
    is_ocl = hasattr(cfg.EXTERNAL.OCL, 'ALGO') and cfg.EXTERNAL.OCL.ALGO != 'PLAIN'
    task_incremental = get_config_attr(cfg, 'EXTERNAL.OCL.TASK_INCREMENTAL', default=False)

    cfg.TUNE = tune

    algo = cfg.EXTERNAL.OCL.ALGO
    if hasattr(cfg,'MNIST'):
        if cfg.MNIST.TASK == 'split':
            goal = 'split_mnist'
        elif cfg.MNIST.TASK == 'permute':
            goal = 'permute_mnist'
        elif cfg.MNIST.TASK == 'rotate':
            goal = 'rotated_mnist'
    if hasattr(cfg, 'CIFAR'):
        goal = 'split_cifar'
        if get_config_attr(cfg, 'CIFAR.DATASET', default="") == 'CIFAR100':
            goal = 'split_cifar100'
        if get_config_attr(cfg, 'CIFAR.MINI_IMAGENET', default=0):
            goal = 'split_mini_imagenet'



    if hasattr(cfg,'MNIST'):

        num_of_datasets = 1 if not task_incremental else get_config_attr(cfg, 'EXTERNAL.OCL.TASK_NUM', totype=int)
        num_of_classes = 10 if not task_incremental else get_config_attr(cfg, 'EXTERNAL.OCL.CLASS_NUM', totype=int)
        base_model = mnist_simple_net_400width_classlearning_1024input_10cls_1ds(num_of_datasets=num_of_datasets,
                                                                                 num_of_classes=num_of_classes,
                                                                                 task_incremental=task_incremental)

        base_model.cfg = cfg
    elif hasattr(cfg, 'CIFAR'):
        if goal == 'split_cifar':
            num_of_datasets = 1 if not task_incremental else get_config_attr(cfg, 'EXTERNAL.OCL.TASK_NUM', totype=int)
            num_of_classes = 10 if not task_incremental else get_config_attr(cfg, 'EXTERNAL.OCL.CLASS_NUM', totype=int)
        elif goal == 'split_cifar100':
            num_of_datasets = 1 if not task_incremental else get_config_attr(cfg, 'EXTERNAL.OCL.TASK_NUM', totype=int)
            num_of_classes = 100 if not task_incremental else get_config_attr(cfg, 'EXTERNAL.OCL.CLASS_NUM', totype=int)
        elif goal == 'split_mini_imagenet':
            num_of_datasets = 1 if not task_incremental else get_config_attr(cfg, 'EXTERNAL.OCL.TASK_NUM', totype=int)
            num_of_classes = 100 if not task_incremental else get_config_attr(cfg, 'EXTERNAL.OCL.CLASS_NUM', totype=int)


        base_model = ResNetClassifier(cfg, depth='18', mlp=1, ignore_index=-100, num_of_datasets=num_of_datasets,
                                      num_of_classes=num_of_classes, task_incremental=task_incremental, goal=goal)
        base_model.cfg = cfg
    else:
        base_model = ResNetClassifier(cfg)

    device = torch.device(cfg.MODEL.DEVICE)
    base_model.to(device)
    if cfg.EXTERNAL.OPTIMIZER.ADAM:
        optimizer = torch.optim.Adam(
            filter(lambda x: x.requires_grad, base_model.parameters()),
            lr=cfg.SOLVER.BASE_LR, betas=(0.9, 0.999)
        )
    else:
        optimizer = torch.optim.SGD(
            filter(lambda x: x.requires_grad, base_model.parameters()),
            lr=cfg.SOLVER.BASE_LR
        )

    # algorithm specific model wrapper
    x_size = 3 * 2 * base_model.cfg.EXTERNAL.IMAGE_SIZE ** 2 if goal == 'classification' else \
        3 * base_model.cfg.EXTERNAL.IMAGE_SIZE ** 2
    if goal == 'split_mnist' or goal == 'permute_mnist' or goal == 'rotated_mnist': x_size = 28 * 28
    if goal == 'split_cifar' or goal == 'split_cifar100':  x_size = 3 * 32 * 32
    if goal == 'split_mini_imagenet': x_size = 3 * 84 * 84

    if algo == 'ER':
        model = ExperienceReplay(base_model, optimizer, x_size, base_model.cfg, goal)
    elif algo == 'VERX':
        model = ExperienceEvolveApprox(base_model, optimizer, x_size, base_model.cfg, goal)
    elif algo == 'AGEM':
        model = AGEM(base_model, optimizer, x_size, base_model.cfg, goal)
    elif algo == 'naive':
        model = NaiveWrapper(base_model, optimizer, x_size, base_model.cfg, goal)
    model.to(device)

    use_mixed_precision = cfg.DTYPE == "float16"
    arguments = {"iteration": 0, "global_step": 0, "epoch": 0}
    output_dir = cfg.OUTPUT_DIR
    writer = None
    epoch_num = 1
    for e in range(epoch_num):
        print("epoch")

        arguments['iteration'] = 0
        epoch = arguments['epoch']
        if goal == 'split_mnist' or goal == 'permute_mnist' or goal == 'rotated_mnist':
            ocl_train_mnist(model, optimizer, None, device, arguments, writer, epoch, goal, tune=tune)
        elif goal == 'split_cifar' or goal == 'split_cifar100' or goal == 'split_mini_imagenet':
            ocl_train_cifar(model, optimizer, None, device, arguments, writer, epoch, goal, tune=tune)
        else:
            raise NotImplementedError
        arguments['epoch'] += 1

        with open(os.path.join(output_dir, 'model.bin'),'wb') as wf:
            torch.save(model.state_dict(), wf)
        # else:
        #     break
        if is_ocl and hasattr(model, 'dump_reservoir') and args.dump_reservoir:
            model.dump_reservoir(os.path.join(cfg.OUTPUT_DIR, 'mem_dump.pkl'), verbose=args.dump_reservoir_verbose)
    return model

def set_cfg_from_args(args, cfg):
    cfg_params = args.cfg
    if cfg_params is None: return
    for param in cfg_params:
        k, v = param.split('=')
        set_config_attr(cfg, k, v)

def count_params(m: torch.nn.Module, only_trainable: bool = False):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = m.parameters()
    if only_trainable:
        parameters = list(p for p in parameters if p.requires_grad)
    unique = dict((p.data_ptr(), p) for p in parameters).values()
    return sum(p.numel() for p in unique)


def main(args):
    if '%id' in args.name:
        exp_name = args.name.replace('%id', get_exp_id())
    else:
        exp_name = args.name

    combined_cfg = CfgNode(new_allowed=True)
    combined_cfg.merge_from_file(args.config)
    cfg = combined_cfg
    cfg.EXTERNAL.EXPERIMENT_NAME = exp_name
    cfg.SEED = args.seed
    cfg.DEBUG = args.debug

    set_cfg_from_args(args, cfg)

    output_dir = get_config_attr(cfg, 'OUTPUT_DIR', default='')
    if output_dir == '.': output_dir = 'runs/'
    cfg.OUTPUT_DIR = os.path.join(output_dir,
                                  '{}_{}'.format(cfg.EXTERNAL.EXPERIMENT_NAME, cfg.SEED))
    cfg.MODE = 'train'

    # cfg.freeze()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    output_dir = cfg.OUTPUT_DIR

    # save overloaded model config in the output directory
    model = train(cfg, local_rank, distributed, tune=args.tune)

    output_args_path = os.path.join(output_dir, 'args.txt')
    wf = open(output_args_path, 'w')
    wf.write(' '.join(sys.argv))
    wf.close()

def seed_everything(seed):
    '''
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def seed_everything_old(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--name', type=str, default='%id')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seed_old', action='store_true')
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--cfg', nargs='*')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dump_reservoir', action='store_true')
    parser.add_argument('--dump_reservoir_verbose', action='store_true')
    parser.add_argument('--single_word', action='store')

    args = parser.parse_args()
    if not args.seed_old:
        seed_everything(args.seed)
    else:
        seed_everything_old(args.seed)
    n_runs = args.n_runs
    for i in range(n_runs):
        main(args)
