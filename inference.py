# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import h5py

import torch
import json
from tqdm import tqdm
from time import sleep

from config import cfg
from utils.utils import Timer
from dataloader import get_dataloader
from sklearn.metrics import f1_score
from collections import defaultdict
from torch.nn import functional as F


def to_list(tensor):
    return tensor.cpu().numpy().tolist()

def compute_on_dataset(model, data_loader, local_rank, device, timer=None, output_file='debug.h5'):
    model.eval()
    results_dict = {}
    gt_dict = {}
    cpu_device = torch.device("cpu")
    text = "GPU {}".format(local_rank)
    pbar = tqdm(
        total=len(data_loader),
        position=local_rank,
        desc=text,
    )
    with h5py.File(output_file, 'w') as f:
        box_feat_ds = f.create_dataset(
            'bbox_features', shape=(len(data_loader), 150, 1024)
        )

        box_ds = f.create_dataset(
            'bboxes', shape=(len(data_loader), 150, 4)
        )

        num_boxes_ds = f.create_dataset(
            'num_boxes', shape=(len(data_loader), 1)
        )

        img_size_ds = f.create_dataset(
            'image_size', shape=(len(data_loader), 2)
        )
        
        info_dict = {}
    
        for e, out_dict in enumerate(data_loader):
            images = out_dict['images']
            targets = out_dict['gt_bboxes']
            image_ids = out_dict['image_ids']
            info = out_dict['info']

            images = [image.to(device) for image in images]
            targets = [target.to(device) for target in targets]
            if timer:
                timer.tic()
            x, _, output = model(images, targets)
            if timer:
                if not cfg.MODEL.DEVICE == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            assert (x.shape[0] == len(targets[0].bbox))
            output = [o.to(cpu_device) for o in output]
            info_dict[image_ids[0]] = {}
            info_dict[image_ids[0]]['idx'] = e
            info_dict[image_ids[0]]['objects'] = {}
            bboxes = targets[0].bbox
            num_boxes = len(targets[0].bbox)
            for i in range(num_boxes):
                tmp = {}
                tmp = info[0][i]
                tmp['idx'] = i
                info_dict[image_ids[0]]['objects'][info[0][i]['object_id']] = tmp

            '''
            for i in range(5):
                fpn_feat_ds[i][e] = output[i].numpy()
            '''
            box_feat_ds[e, :num_boxes, :] = x.cpu().numpy()
            box_ds[e, :num_boxes, :] = bboxes.cpu().numpy()
            num_boxes_ds[e] = num_boxes
            img_size_ds[e] = targets[0].size
            pbar.update(1)
            sleep(0.001)

    with open(output_file.replace('.h5', '_map.json'), 'w') as fp:
        json.dump(info_dict, fp)


def inference_step(model, out_dict, device=torch.device('cuda')):
    images = torch.stack(out_dict['images'])
    obj_labels = torch.cat(out_dict['object_labels'], -1)
    attr_labels = torch.cat(out_dict['attribute_labels'], -1)
    cropped_image = torch.stack(out_dict['cropped_image'])
    images = images.to(device)
    obj_labels = obj_labels.to(device)
    attr_labels = attr_labels.to(device)

    cropped_image = cropped_image.to(device)
    # loss_dict = model(images, targets)
    ret_dict = model(bbox_images=cropped_image, spatial_feat=None,
                     attr_labels=attr_labels, obj_labels=obj_labels,
                     images=images)
    attr_score, obj_score = ret_dict.get('attr_score', None), \
                            ret_dict.get('obj_score', None)
    if attr_score is not None:
        attr_score_norm = F.softmax(attr_score, -1)
        ret_dict['pred_attr_prob'], ret_dict['pred_attr'] = attr_score_norm.max(-1)
    if obj_score is not None:
        obj_score_norm = F.softmax(obj_score, -1)
        ret_dict['pred_obj_prob'], ret_dict['pred_obj'] = obj_score_norm.max(-1)
    ret_dict['obj_labels'], ret_dict['attr_labels'] = obj_labels, attr_labels
    return ret_dict


def inference(
        model,
        current_epoch,
        current_iter,
        local_rank,
        data_loader,
        dataset_name,
        device="cuda",
        max_instance=3200,
        mute=False,
        verbose_return=False
):
    model.train(False)
    # convert to a torch.device for efficiency
    device = torch.device(device)
    if not mute:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.info("Start evaluation")
    total_timer = Timer()
    total_timer.tic()
    torch.cuda.empty_cache()
    if not mute:
        pbar = tqdm(
            total=len(data_loader),
            desc="Validation in progress"
        )

    def to_list(tensor):
        return tensor.cpu().numpy().tolist()
    with torch.no_grad():
        all_pred_obj, all_truth_obj, all_pred_attr, all_truth_attr = [], [], [], []
        all_image_ids, all_boxes = [], []
        all_pred_attr_prob = []
        all_raws = []
        obj_loss_all, attr_loss_all = 0, 0

        cnt = 0
        for iteration, out_dict in enumerate(data_loader):
            if type(max_instance) is int:
                if iteration == max_instance // model.cfg.EXTERNAL.BATCH_SIZE: break
            if type(max_instance) is float:
                if iteration > max_instance * len(data_loader) // model.cfg.EXTERNAL.BATCH_SIZE: break
            # print(iteration)

            if verbose_return:
                all_image_ids.extend(out_dict['image_ids'])
                all_boxes.extend(out_dict['gt_bboxes'])
                all_raws.extend(out_dict['raw'])

            ret_dict = inference_step(model, out_dict, device)
            loss_attr, loss_obj, attr_score, obj_score = ret_dict.get('attr_loss', None), \
                                                         ret_dict.get('obj_loss', None), \
                                                         ret_dict.get('attr_score', None), \
                                                         ret_dict.get('obj_score', None)

            if loss_attr is not None:
                attr_loss_all += loss_attr.item()
                pred_attr_prob, pred_attr = ret_dict['pred_attr_prob'], ret_dict['pred_attr']
                all_pred_attr.extend(to_list(pred_attr))
                all_truth_attr.extend(to_list(ret_dict['attr_labels']))
                all_pred_attr_prob.extend(to_list(pred_attr_prob))
            if loss_obj is not None:
                obj_loss_all += loss_obj.item()
                _, pred_obj = obj_score.max(-1)
                all_pred_obj.extend(to_list(pred_obj))
                all_truth_obj.extend(to_list(ret_dict['obj_labels']))
            cnt += 1
            if not mute:
                pbar.update(1)

        obj_f1 = f1_score(all_truth_obj, all_pred_obj, average='micro')
        attr_f1 = f1_score(all_truth_attr, all_pred_attr, average='micro')
        obj_loss_all /= (cnt + 1e-10)
        attr_loss_all /= (cnt + 1e-10)
        if not mute:
            logger.info('Epoch: {}\tIteration: {}\tObject f1: {}\tAttr f1:{}\tObject loss:{}\tAttr loss:{}'.
                    format(current_epoch, current_iter, obj_f1, attr_f1, obj_loss_all, attr_loss_all))
        #compute_on_dataset(model, data_loader, local_rank, device, inference_timer, output_file)
    # wait for all processes to complete before measuring the time
    total_time = total_timer.toc()
    model.train(True)
    if not verbose_return:
        return obj_f1, attr_f1, len(all_truth_attr)
    else:
        return obj_f1, attr_f1, all_pred_attr, all_truth_attr, all_pred_obj, all_truth_obj, all_image_ids, all_boxes, \
               all_pred_attr_prob, all_raws

def run_forget_metrics(metric_dict, finished_tasks, all_tasks, key, forget_dict):
    for task in all_tasks:
        if len(metric_dict[key][task]) <= 1 or task not in finished_tasks:
            forget_dict[task].append(-1)
        else:
            forget_dict[task].append(max(metric_dict[key][task][:-1]) - metric_dict[key][task][-1])


def run_forward_transfer_metrics(metric_dict, seen_tasks, all_tasks, key, ft_dict):
    for task in all_tasks:
        if task not in seen_tasks:
            ft_dict[task].append(metric_dict[key][task][-1])
        else:
            ft_dict[task].append(-1)


def numericalize_metric_scores(metric_dict):
    result_dict = defaultdict(list)
    for t in range(metric_dict['length']):
        for key in ['attr_acc', 'forget_dict', 'obj_acc']:
            total_inst = 0
            total_score = 0
            for attr in metric_dict[key]:
                inst_num = metric_dict['inst_num'][attr][t]
                score = metric_dict[key][attr][t]
                if score != -1:
                    total_inst += inst_num
                    total_score += score * inst_num
            avg_score = total_score / (total_inst + 1e-10)
            result_dict[key].append(avg_score)
    return result_dict


def inference_ocl_attr(
    model,
    current_epoch,
    current_iter,
    dataset_name,
    prev_metric_dict,
    seen_objects,
    finished_objects,
    all_objects,
    max_instance
):
    """

    :param model:
    :param current_epoch:
    :param current_iter:
    :param prev_metric_dict: {attr_acc: <attr>: [acc1, acc2]}
    :param filter_objects:
    :param filter_attrs:
    :return:
    """
    model.train(False)
    device = torch.device('cuda')

    if not prev_metric_dict:
        prev_metric_dict = {
            'attr_acc': defaultdict(list),
            'inst_num': defaultdict(list),
            'ft_dict': defaultdict(list),
            'forget_dict': defaultdict(list),
            'obj_acc': defaultdict(list),
            'length': 0
        }

    pbar = tqdm(
        total=len(all_objects),
        desc="Validation in progress"
    )
    # only seen objects by this time
    for obj in all_objects:
        dataloader = get_dataloader(model.cfg, 'val',False,False,filter_obj=[obj])
        obj_acc, attr_acc, inst_num = inference(model, current_epoch, current_iter, 0, dataloader, dataset_name,
                                                max_instance=max_instance, mute=True)

        prev_metric_dict['attr_acc'][obj].append(attr_acc)
        prev_metric_dict['inst_num'][obj].append(inst_num)
        prev_metric_dict['obj_acc'][obj].append(obj_acc)
        pbar.update(1)

    metric_dict = prev_metric_dict
    #run_forward_transfer_metrics(metric_dict, seen_objects, all_objects, 'attr_acc', metric_dict['ft_dict'])
    run_forget_metrics(metric_dict, finished_objects, all_objects, 'attr_acc', metric_dict['forget_dict'])
    metric_dict['length'] += 1

    numerical_metric_dict = numericalize_metric_scores(metric_dict)

    return metric_dict, numerical_metric_dict

def inference_mean_exemplar(
        model,
        current_epoch,
        current_iter,
        local_rank,
        data_loader,
        dataset_name,
        device="cuda",
        max_instance=3200,
        mute=False,
):
    model.train(False)
    # convert to a torch.device for efficiency
    device = torch.device(device)
    if not mute:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.info("Start evaluation")
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    torch.cuda.empty_cache()
    if not mute:
        pbar = tqdm(
            total=len(data_loader),
            desc="Validation in progress"
        )
    with torch.no_grad():
        all_pred_obj, all_truth_obj, all_pred_attr, all_truth_attr = [], [], [], []
        obj_loss_all, attr_loss_all = 0, 0
        cnt = 0
        for iteration, out_dict in enumerate(data_loader):
            if type(max_instance) is int:
                if iteration == max_instance // model.cfg.EXTERNAL.BATCH_SIZE: break
            if type(max_instance) is float:
                if iteration > max_instance * len(data_loader) // model.cfg.EXTERNAL.BATCH_SIZE: break
            # print(iteration)
            images = torch.stack(out_dict['images'])
            obj_labels = torch.cat(out_dict['object_labels'], -1)
            attr_labels = torch.cat(out_dict['attribute_labels'], -1)
            cropped_image = torch.stack(out_dict['cropped_image'])

            images = images.to(device)
            obj_labels = obj_labels.to(device)
            attr_labels = attr_labels.to(device)

            cropped_image = cropped_image.to(device)
            # loss_dict = model(images, targets)
            pred_obj = model.mean_of_exemplar_classify(cropped_image)

            all_pred_obj.extend(to_list(pred_obj))
            all_truth_obj.extend(to_list(obj_labels))
            cnt += 1
            if not mute:
                pbar.update(1)

        obj_f1 = f1_score(all_truth_obj, all_pred_obj, average='micro')
        #attr_f1 = f1_score(all_truth_attr, all_pred_attr, average='micro')
        obj_loss_all /= (cnt + 1e-10)
    # wait for all processes to complete before measuring the time
    total_time = total_timer.toc()
    model.train(True)
    return obj_f1, 0, len(all_truth_obj)