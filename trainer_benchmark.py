#from trainer import *
import logging
import torch

from tqdm import tqdm
import json, os

from inference import to_list, f1_score
from dataloader import get_split_mnist_dataloader, get_permute_mnist_dataloader,\
                    get_split_cifar_dataloader, get_split_cifar100_dataloader, get_split_mini_imagenet_dataloader,\
                    get_rotated_mnist_dataloader, IIDDataset
from utils.utils import get_config_attr
from torch.utils.data import DataLoader

def exp_decay_lr(optimizer, step, total_step, init_lr):
    gamma = (1 / 6) ** (step / total_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * gamma

def ocl_train_mnist(model, optimizer, checkpointer, device, arguments, writer, epoch,
                    goal='split', tune=False):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training @ epoch {:02d}".format(arguments['epoch']))
    model.train()
    cfg = model.cfg
    pbar = tqdm(
        position=0,
        desc='GPU: 0'
    )

    num_instances = cfg.MNIST.INSTANCE_NUM

    if goal == 'split_mnist':
        task_num = 5
        loader_func = get_split_mnist_dataloader
    elif goal == 'permute_mnist':
        task_num = 10
        loader_func = get_permute_mnist_dataloader
    elif goal == 'rotated_mnist':
        task_num = 20
        loader_func = get_rotated_mnist_dataloader
    else:
        raise ValueError

    if tune:
        task_num = get_config_attr(cfg, 'EXTERNAL.OCL.TASK_NUM', totype=int, default=3)

    num_epoch = get_config_attr(cfg, 'EXTERNAL.EPOCH', totype=int, default=1)
    total_step = task_num * 1000
    base_lr = get_config_attr(cfg,'SOLVER.BASE_LR',totype=float)
    # whether iid
    iid = not get_config_attr(cfg, 'EXTERNAL.OCL.ACTIVATED', totype=bool)
    do_exp_lr_decay = get_config_attr(cfg,'EXTERNAL.OCL.EXP_LR_DECAY',0)

    all_accs = []
    best_avg_accs = []
    step = 0
    for task_id in range(task_num):
        if iid:
            if task_id != 0: break
            data_loaders = [loader_func(cfg, 'train', [task_id], batch_size=cfg.EXTERNAL.BATCH_SIZE,
                                      max_instance=num_instances) for task_id in range(task_num)]
            data_loader = DataLoader(IIDDataset(data_loaders), batch_size=cfg.EXTERNAL.BATCH_SIZE)
            num_instances *= task_num
        else:
            data_loader = loader_func(cfg, 'train', [task_id], batch_size=cfg.EXTERNAL.BATCH_SIZE,
                                      max_instance=num_instances)

        best_avg_acc = -1
        #model.net.set_task(task_id) # choose the classifier head if the model supports
        for epoch in range(num_epoch):
            seen = 0
            for i, data in enumerate(data_loader):
                if seen >= num_instances: break
                inputs, labels = data
                inputs, labels = (inputs.to(device), labels.to(device))
                task_ids = torch.LongTensor([task_id] * labels.size(0)).to(inputs.device)
                inputs = inputs.flatten(1)
                model.observe(inputs, labels, task_ids=task_ids)
                step += 1
                if do_exp_lr_decay:
                    exp_decay_lr(optimizer, step, total_step, base_lr)

                seen += labels.size(0)
            # run evaluation
            with torch.no_grad():
                if iid:
                    accs, _, avg_acc = inference_mnist(model, task_num, loader_func, device, tune=tune)
                else:
                    accs, _, avg_acc = inference_mnist(model, task_id + 1, loader_func, device, tune=tune)
            logger.info('Epoch {}\tTask {}\tAcc {}'.format(epoch, task_id, avg_acc))
            for i, acc in enumerate(accs):
                logger.info('::Val Task {}\t Acc {}'.format(i, acc))
            all_accs.append(accs)
            if avg_acc > best_avg_acc:
                best_avg_acc = avg_acc
            else:
                break
        best_avg_accs.append(best_avg_acc)
    file_name = 'result.json' if not tune else 'result_tune_k{}.json'.format(task_num)
    result_file = open(os.path.join(cfg.OUTPUT_DIR, file_name), 'w')
    json.dump({'all_accs': all_accs, 'avg_acc': avg_acc}, result_file, indent=4)
    result_file.close()


def inference_mnist(model, max_task, loader_func, device, tune=False):
    model.train(False)
    accs, instance_nums = [], []
    for val_task_id in range(0, max_task):
        #task_id = 0
        all_pred, all_truth = [], []
        val_data_loader = loader_func(model.cfg, 'test' if not tune else 'val', [val_task_id],
                                      batch_size=model.cfg.EXTERNAL.BATCH_SIZE)
        print('-------len val data loader {}-------'.format(len(val_data_loader)))
        for i, data in enumerate(val_data_loader):

            inputs, labels = data
            inputs, labels = (inputs.to(device), labels.to(device))
            task_ids = torch.LongTensor([val_task_id] * labels.size(0)).to(inputs.device)
            ret_dict = model(inputs, labels, task_ids=task_ids)
            score = ret_dict['score']
            _, pred = torch.max(score, -1)
            all_pred.extend(to_list(pred))
            all_truth.extend(to_list(labels))
        acc = f1_score(all_truth, all_pred, average='micro')
        accs.append(acc)
        instance_nums.append(len(all_pred))
    total_instance_num = sum(instance_nums)
    model.train(True)
    return accs, instance_nums, sum([x * y / total_instance_num for x,y in zip(accs, instance_nums)])


def ocl_train_cifar(model, optimizer, checkpointer, device, arguments, writer, epoch,
                    goal='split_cifar', tune=False):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training @ epoch {:02d}".format(arguments['epoch']))
    model.train()
    cfg = model.cfg

    num_epoch = cfg.CIFAR.EPOCH
    if goal == 'split_cifar':
        loader_func = get_split_cifar_dataloader
        total_step = 4750
    elif goal == 'split_cifar100':
        loader_func = get_split_cifar100_dataloader
        total_step = 25000
    else:
        loader_func = get_split_mini_imagenet_dataloader
        total_step = 22500
    max_instance = cfg.CIFAR.INSTANCE_NUM if hasattr(cfg.CIFAR, 'INSTANCE_NUM') else 1e10
    if not tune:
        task_num = get_config_attr(cfg, 'EXTERNAL.OCL.TASK_NUM', totype=int)
    else:
        task_num = get_config_attr(cfg, 'EXTERNAL.OCL.TASK_NUM', totype=int)


    do_exp_lr_decay = get_config_attr(cfg,'EXTERNAL.OCL.EXP_LR_DECAY',0)
    base_lr = get_config_attr(cfg,'SOLVER.BASE_LR',totype=float)
    step = 0

    num_epoch = get_config_attr(cfg, 'EXTERNAL.EPOCH', totype=int, default=1)
    all_accs = []
    best_avg_accs = []
    iid = not get_config_attr(cfg, 'EXTERNAL.OCL.ACTIVATED', totype=bool)
    for task_id in range(task_num):
        if iid:
            if task_id != 0: break
            data_loaders = [loader_func(cfg, 'train', [task_id], batch_size=cfg.EXTERNAL.BATCH_SIZE,
                                        max_instance=max_instance) for task_id in range(task_num)]
            data_loader = DataLoader(IIDDataset(data_loaders), batch_size=cfg.EXTERNAL.BATCH_SIZE)
            max_instance *= task_num
        else:
            data_loader = loader_func(cfg, 'train', [task_id], batch_size=cfg.EXTERNAL.BATCH_SIZE, max_instance=max_instance)
        pbar = tqdm(
            position=0,
            desc='GPU: 0',
            total=len(data_loader)
        )
        best_avg_acc = -1
        for epoch in range(num_epoch):
            seen = 0
            for i, data in enumerate(data_loader):
                if seen >= max_instance: break
                pbar.update(1)
                inputs, labels = data
                inputs, labels = (inputs.to(device), labels.to(device))
                inputs = inputs.flatten(1)
                task_ids = torch.LongTensor([task_id] * labels.size(0)).to(inputs.device)
                model.observe(inputs, labels, task_ids)
                seen += inputs.size(0)
                if do_exp_lr_decay:
                    exp_decay_lr(optimizer, step, total_step, base_lr)
                step += 1
            # # run evaluation
            with torch.no_grad():
                if iid:
                    accs, _, avg_acc = inference_cifar(model, task_num, loader_func, device, goal, tune=tune)
                else:
                    accs, _, avg_acc = inference_cifar(model, task_id + 1, loader_func, device, goal, tune=tune)
            logger.info('Epoch {}\tTask {}\tAcc {}'.format(epoch, task_id, avg_acc))
            for i, acc in enumerate(accs):
                logger.info('::Val Task {}\t Acc {}'.format(i, acc))
            all_accs.append(accs)
            if avg_acc > best_avg_acc:
                best_avg_acc = avg_acc
            else:
                break
        best_avg_accs.append(best_avg_acc)
    file_name = 'result.json' if not tune else 'result_tune_k{}.json'.format(task_num)
    result_file = open(os.path.join(cfg.OUTPUT_DIR, file_name), 'w')
    json.dump({'all_accs': all_accs, 'avg_acc': avg_acc, 'best_avg_accs': best_avg_accs}, result_file, indent=4)
    result_file.close()
    return best_avg_accs

def inference_cifar(model, max_task, loader_func, device, goal, tune=False):
    accs, instance_nums = [], []
    model.train(False)
    for val_task_id in range(0, max_task):
        all_pred, all_truth = [], []
        val_data_loader = loader_func(model.cfg, 'test' if not tune else 'val', [val_task_id], batch_size=model.cfg.EXTERNAL.BATCH_SIZE)
        for i, data in enumerate(val_data_loader):
            inputs, labels = data
            inputs, labels = (inputs.to(device), labels.to(device))
            inputs = inputs.view(-1, 3, 32, 32) if goal == 'split_cifar' or goal == 'split_cifar100' else inputs.view(-1, 3, 84, 84)
            task_ids = torch.LongTensor([val_task_id] * labels.size(0)).to(inputs.device)
            if model.cfg.EXTERNAL.OCL.ALGO == 'CNDPM':
                score = model(inputs)
            else:
                ret_dict = model(bbox_images=inputs, spatial_feat=None, attr_labels=labels,
                                obj_labels=None, images=None, task_ids=task_ids)
                score = ret_dict['score']
            _, pred = torch.max(score, -1)
            all_pred.extend(to_list(pred))
            all_truth.extend(to_list(labels))
        acc = f1_score(all_truth, all_pred, average='micro')
        accs.append(acc)
        instance_nums.append(len(all_pred))
    total_instance_num = sum(instance_nums)
    model.train(True)
    return accs, instance_nums, sum([x * y / total_instance_num for x,y in zip(accs, instance_nums)])


