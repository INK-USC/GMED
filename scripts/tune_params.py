import os
import logging
import argparse
import random

logger = logging.getLogger(__name__)

ds_to_name = {'split_cifar': 'cifar10_5tasks_2class_ci_verx', 'mini_imagenet': 'mini_imagenet_ci_verx',
              'rotated_mnist': 'rotated_mnist_verx', 'split_mnist':'split_mnist_verx',
              'permuted_mnist': 'pm_verx'}
ds_to_config = {'split_cifar': 'verx_cifar.yaml', 'mini_imagenet': 'verx_mini_imagenet.yaml',
                'rotated_mnist': 'ver_rotate_approx.yaml', 'split_mnist':'ver_approx.yaml',
                'permuted_mnist': 'ver_permute_approx.yaml'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--stop_seed', type=int, default=5)
    parser.add_argument('--dataset')
    parser.add_argument('--strides', type=float, nargs='+', default=[0.1,0.05,0.5,1.0])
    parser.add_argument('--proj_flags', type=int, nargs='+', default=[2])
    parser.add_argument('--reg_strengths', type=float, nargs='+', default=[0.001,0.01,0.1])
    parser.add_argument('--lr', type=float, nargs='*', default=[0])
    parser.add_argument('--iters', type=int, nargs='+')
    parser.add_argument('--mem_sizes', type=int, nargs='+')
    parser.add_argument('--use_relu', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--task_num', type=int, default=3)
    parser.add_argument('--extra', default='')
    parser.add_argument('--name', default='')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--mir', action='store_true')
    parser.add_argument('--rand', action='store_true')

    args = parser.parse_args()

    logger.addHandler(logging.FileHandler('logs/{}_{}.txt'.format(args.dataset, random.randint(0,10000000))))
    logger.setLevel(logging.INFO)

    strides = args.strides
    mem_sizes = args.mem_sizes
    iters = args.iters

    use_loss_1 = 1 if not args.rand else -2

    extra_auto = ''
    if args.mir:
        extra_auto = 'EXTERNAL.OCL.MIR=1 EXTERNAL.REPLAY.MEM_BS=10 EXTERNAL.REPLAY.MIR_K=10 EXTERNAL.OCL.EDIT_RANDOM=1'

    for stride in strides:
        for mem_size in mem_sizes:
            for proj_flag in args.proj_flags:
                for reg_strength in args.reg_strengths:
                    for lr in args.lr:

                        for iter_n in iters:
                            for seed in range(args.start_seed, args.stop_seed):
                                name_mark = ''
                                if args.use_relu:
                                    name_mark += 'relu'
                                if args.mir:
                                    name_mark += 'mir'

                                name = '{}_iter{}_stride_{}_{}_0_{}{}_m{}_{}_{}_{}'.format(ds_to_name[args.dataset], iter_n, stride, use_loss_1,
                                                                                       proj_flag, reg_strength, mem_size, seed,
                                                                                        name_mark,
                                                                                        args.name)
                                if lr != 0:
                                    name += '_lr{}_epoch{}'.format(lr, args.epoch)
                                result_file = 'runs/{}_{}/result_tune_k{}.json'.format(name, seed, args.task_num)
                                if not args.overwrite and os.path.isfile(result_file):
                                    print('result file exists, skiping')
                                else:
                                    command = 'python train.py --name {name} --tune --config configs/memevolve/{config} --seed {seed} ' \
                                              '--cfg EXTERNAL.OCL.GRAD_ITER={grad_iter} EXTERNAL.OCL.GRAD_STRIDE={grad_stride} '\
                                              'EXTERNAL.OCL.USE_LOSS_1={use_loss_1} EXTERNAL.OCL.USE_LOSS_2=0 EXTERNAL.OCL.USE_RELU=0 '\
                                              'EXTERNAL.OCL.PROJ_LOSS_REG={proj_flag} EXTERNAL.REPLAY.MEM_LIMIT={mem_size} ' \
                                              'EXTERNAL.OCL.REG_STRENGTH={reg_strength} ' \
                                              'EXTERNAL.OCL.USE_RELU={use_relu} ' \
                                              'EXTERNAL.EPOCH={epoch} ' \
                                              'EXTERNAL.BATCH_SIZE={batch_size} ' \
                                              'EXTERNAL.OCL.TASK_NUM={task_num} ' \
                                              '{extra} ' \
                                              '{extra_auto} '\
                                              .format(**{'name': name, 'config': ds_to_config[args.dataset], 'seed':seed,
                                                       'grad_iter':iter_n, 'grad_stride': stride, 'mem_size': mem_size,
                                                       'extra': args.extra, 'proj_flag': proj_flag, 'reg_strength': reg_strength,
                                                     'use_relu': args.use_relu, 'epoch': args.epoch, 'extra_auto': extra_auto,
                                                     'batch_size': args.batch_size, 'task_num': args.task_num, 'use_loss_1': use_loss_1})
                                    if lr != 0:
                                        command += ' SOLVER.BASE_LR={} '.format(lr)
                                    logger.info(command)
                                    exit_code = os.system(command)
                                    logger.info('Exit code {}'.format(exit_code))