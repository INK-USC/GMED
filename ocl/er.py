import torch
import numpy as np
import pickle
from utils.utils import get_config_attr
import copy
from .naive import NaiveWrapper
from torch.optim import SGD, Adam
from collections import defaultdict


import math
try:
    from pytorch_transformers import AdamW
except ImportError:
    AdamW = None


def y_to_np(y):
    if type(y) is tuple:
        return tuple(x.item() for x in y)
    else:
        return y.cpu().numpy()


def y_to_cpu(y):
    if torch.is_tensor(y):
        y = y.cpu()
    else:
        y = [_.cpu() for _ in y]
    return y


def index_select(l, indices, device):
    ret = []
    for i in indices:
        if type(l[i]) is np.ndarray:
            x = torch.from_numpy(l[i]).to(device)
            ret.append(x.unsqueeze(0))
        else:
            if type(l[i]) is list:
                item = []
                for j in range(len(l[i])):
                    if type(l[i][j]) is np.ndarray:
                        item.append(torch.from_numpy(l[i][j]))
                    else:
                        item.append(l[i][j])
                ret.append(item)
            else:
                ret.append(l[i])
    return ret


def concat_with_padding(l):
    if l is None or l[0] is None: return None
    if type(l[0]) in [list, tuple]:
        ret = [torch.cat(t, 0) for t in zip(*l)]
    else:
        if len(l[0].size()) == 2:  # potentially requires padding
            max_length = max([x.size(1) for x in l])
            ret = []
            for x in l:
                pad = torch.zeros(x.size(0), max_length - x.size(1)).long().to(x.device)
                x_pad = torch.cat([x, pad], -1)
                ret.append(x_pad)
            ret = torch.cat(ret, 0)
        else:
            ret = torch.cat(l, 0)
    return ret


def store_grad(pp, grads, grad_dims):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads.fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en].copy_(param.grad.data.view(-1))
        cnt += 1

class ExperienceReplay(NaiveWrapper):
    def __init__(self, base, optimizer, input_size, cfg, goal):
        super().__init__(base, optimizer, input_size, cfg, goal)
        self.net = base
        self.optimizer = optimizer
        self.mem_limit = cfg.EXTERNAL.REPLAY.MEM_LIMIT
        self.mem_bs = cfg.EXTERNAL.REPLAY.MEM_BS
        self.input_size = input_size
        self.reservoir, self.example_seen = None, None
        self.reset_mem()
        self.mem_occupied = {}

        self.seen_tasks = []
        self.balanced = False
        self.policy = get_config_attr(cfg, 'EXTERNAL.OCL.POLICY', default='reservoir', totype=str)
        self.mir_k = get_config_attr(cfg, 'EXTERNAL.REPLAY.MIR_K', default=10, totype=int)
        self.mir = get_config_attr(cfg, 'EXTERNAL.OCL.MIR', default=0, totype=int)

        self.mir_agg = get_config_attr(cfg, 'EXTERNAL.OCL.MIR_AGG', default='avg', totype=str)

        self.concat_replay = get_config_attr(cfg, 'EXTERNAL.OCL.CONCAT', default=0, totype=int)
        self.separate_replay = get_config_attr(cfg, 'EXTERNAL.OCL.SEPARATE', default=0, totype=int)
        self.mem_augment = get_config_attr(cfg, 'EXTERNAL.OCL.MEM_AUG', default=0, totype=int)
        self.legacy_aug = get_config_attr(cfg, 'EXTERNAL.OCL.LEGACY_AUG', default=0, totype=int)
        self.use_hflip_aug = get_config_attr(cfg,'EXTERNAL.OCL.USE_HFLIP_AUG',default=1,totype=int)
        self.padding_aug = get_config_attr(cfg,'EXTERNAL.OCL.PADDING_AUG',default=-1,totype=int)
        self.rot_aug = get_config_attr(cfg,'EXTERNAL.OCL.ROT_AUG',default=-1,totype=int)

        self.lb_reservoir = get_config_attr(cfg,'EXTERNAL.OCL.LB_RESERVOIR', default=0)

        self.cfg = cfg
        self.grad_dims = []

        for param in self.parameters():
            self.grad_dims.append(param.data.numel())

    def get_config_padding(self, pad, rot):
        if self.padding_aug != -1:
            pad = self.padding_aug
        if self.rot_aug != -1:
            rot = self.rot_aug
        return pad, rot

    def reset_mem(self):
        self.reservoir = {'x': np.zeros((self.mem_limit, self.input_size)),
                          'y': [None] * self.mem_limit,
                          'y_extra': [None] * self.mem_limit,
                          'replay_time': np.zeros(self.mem_limit),
                          'loss_stats': np.zeros(self.mem_limit),
                          'label_cnts': defaultdict(int),
                          }
        self.example_seen = 0

    def update_mem(self, *args, **kwargs):
        if self.policy == 'balanced':
            return self.update_mem_balanced(*args, **kwargs)
        elif self.policy == 'reservoir':
            return self.update_mem_reservoir(*args, **kwargs)
        elif self.policy == 'clus':
            return self.update_mem_kmeans(*args, **kwargs)
        else:
            raise ValueError

    def reinit_mem(self, xsize):
        self.input_size = xsize
        self.reset_mem()

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def update_mem_reservoir(self, x, y, y_extra=None, loss_x=None, *args, **kwargs):
        if self.example_seen == 0 and self.reservoir['x'].shape[-1] != x.shape[-1]:
            self.reinit_mem(x.shape[-1])

        x = x.cpu().numpy()
        if type(y) not in [list, tuple]:
            y = y_to_np(y)
            #if y.shape == (1,):
            #    y = y[0]
        else:
            y = y_to_cpu(y)

        if type(y_extra) not in [list, tuple]:
            y_extra = y_to_np(y_extra)
        elif y_extra is not None:
            y_extra = y_to_cpu(y_extra)

        if self.example_seen < self.mem_limit:
            self.reservoir['x'][self.example_seen] = x
            self.reservoir['y'][self.example_seen] = y
            self.reservoir['y_extra'][self.example_seen] = y_extra
            #self.reservoir['replay_time'][self.example_seen] = 0
            self.reservoir['label_cnts'][y.item()] += 1
            j = self.example_seen
        else:

            #
            #else:
            j = np.random.RandomState(self.example_seen + self.cfg.SEED).randint(0, self.example_seen)
            if j < self.mem_limit:
                if self.lb_reservoir:
                    j = self.get_loss_aware_balanced_reservoir_sampling_index()
                self.reservoir['label_cnts'][self.reservoir['y'][j].item()] -= 1
                self.reservoir['x'][j] = x
                self.reservoir['y'][j] = y
                self.reservoir['y_extra'][j] = y_extra
                #self.reservoir['replay_time'][j] = 0
                self.reservoir['label_cnts'][y.item()] += 1

        #if loss_x is not None:
        #    self.reservoir['loss_stats'][j] = loss_x if not torch.is_tensor(loss_x) else loss_x.item()
            #self.reservoir['loss_stat_steps'][j] = [self.example_seen]
            #self.reservoir['forget'] = 0
        self.example_seen += 1

    def update_loss_states(self, loss, indices):
        for i in range(len(indices)):
            self.reservoir['loss_stats'][indices[i]] = loss[i].item()

    def get_loss_aware_balanced_reservoir_sampling_index(self):
        # assumes the mem is full
        _random = np.random.RandomState(self.example_seen + self.cfg.SEED)
        s_balance = np.array([self.reservoir['label_cnts'][mem_y.item()] for mem_y in self.reservoir['y']])
        s_loss = -self.reservoir['loss_stats']
        alpha = abs(s_balance.sum()) / s_loss.sum()
        s = s_loss * alpha + s_balance
        probs = s / s.sum()
        idx = _random.choice(len(probs), p=probs)
        return idx

    def update_mem_kmeans(self, x, y, y_extra=None, x_feat=None, **kwargs):
        if 'x_clus' not in self.reservoir:
            self.reservoir['x_clus'] = np.zeros((self.mem_limit, x_feat.shape[0]))
            self.reservoir['x_cnt'] = [None] * self.mem_limit
            self.reservoir['x_feat'] = np.zeros((self.mem_limit, x_feat.shape[0]))
        x = x.cpu().numpy()

        if type(y) not in [list, tuple]:
            y = y_to_np(y)
        else:
            y = y_to_cpu(y)

        if self.example_seen < self.mem_limit:
            self.reservoir['x'][self.example_seen] = x
            self.reservoir['y'][self.example_seen] = y
            self.reservoir['y_extra'][self.example_seen] = y_extra
            self.reservoir['x_clus'][self.example_seen] = x_feat
            self.reservoir['x_cnt'][self.example_seen] = 0
            self.reservoir['x_feat'][self.example_seen] = x_feat
        else:
            # compute L2 distance
            center_i, center_d = -1, 1e10
            for mb in range(self.mem_limit):
                dist = np.sum(np.square(x_feat - self.reservoir['x_clus'][mb]))
                if dist < center_d:
                    center_i = mb
                    center_d = dist
            cnt = self.reservoir['x_cnt'][center_i]
            self.reservoir['x_clus'][center_i] = (self.reservoir['x_clus'][center_i] * cnt + x_feat) / (cnt + 1)
            self.reservoir['x_cnt'][center_i] += 1

            d_mem = np.sum(np.square(x_feat - self.reservoir['x_clus'][center_i]))
            d_mem_best = np.sum(np.square(self.reservoir['x_feat'][center_i] - self.reservoir['x_clus'][center_i]))
            if d_mem < d_mem_best:
                self.reservoir['x'][center_i] = x
                self.reservoir['y'][center_i] = y
                self.reservoir['y_extra'][center_i] = y_extra
                self.reservoir['x_feat'][center_i] = x_feat
        self.example_seen += 1

    def compute_offset(self, task_id, n):
        idx = self.seen_tasks.index(task_id)
        return int(idx / n * self.mem_limit), int((idx + 1) / n * self.mem_limit)

    def reallocate_memory(self, old_mem, old_occ):
        new_mem = {'x': np.zeros((self.mem_limit, self.input_size)),
                   'y': [None] * self.mem_limit,
                   'y_extra': [None] * self.mem_limit}
        new_occ = {}
        n_tasks = len(self.seen_tasks)
        for task in self.seen_tasks:
            old_offset_start, old_offset_stop = self.compute_offset(task, n_tasks)
            new_offset_start, new_offset_stop = self.compute_offset(task, n_tasks + 1)
            i = 0
            while new_offset_start + i < new_offset_stop:
                new_offset = new_offset_start + i
                old_offset = old_offset_start + i
                for k in old_mem:
                    new_mem[k][new_offset] = old_mem[k][old_offset]
                i += 1
            new_occ[task] = old_occ[task]  # min(old_occ[task], new_offset_stop - new_offset_start)
        return new_mem, new_occ

    def update_mem_balanced(self, x, y):
        y_attr, y_obj = y_to_np(y)
        y_obj = y_obj.item()
        y_attr = y_attr.item()
        x = x.cpu().numpy()

        if y_obj not in self.seen_tasks:
            # reallocate memory by expanding seen task by 1
            new_mem, new_occ = self.reallocate_memory(self.reservoir, self.mem_occupied)
            self.reservoir = new_mem
            self.mem_occupied = new_occ
            self.mem_occupied[y_obj] = 0
            self.seen_tasks.append(y_obj)
        offset_start, offset_stop = self.compute_offset(y_obj, len(self.seen_tasks))

        if self.mem_occupied[y_obj] < offset_stop - offset_start:
            pos = self.mem_occupied[y_obj] + offset_start
            self.reservoir['x'][pos] = x
            self.reservoir['y'][pos] = y
            # self.reservoir['y_attr'][pos] = y_attr
        else:
            j = np.random.RandomState(self.example_seen + self.cfg.SEED).randint(0, self.mem_occupied[y_obj])
            if j < offset_stop - offset_start:
                pos = j + offset_start
                self.reservoir['x'][pos] = x
                self.reservoir['y'][pos] = y
                # self.reservoir['y_attr'][pos] = y_attr

        self.mem_occupied[y_obj] += 1
        self.example_seen += 1

    def get_available_index(self):
        l = []
        for idx, t in enumerate(self.seen_tasks):
            offset_start, offset_stop = self.compute_offset(t, len(self.seen_tasks))
            for i in range(offset_start, min(offset_start + self.mem_occupied[t], offset_stop)):
                l.append(i)
        return l

    def get_random(self, seed=1):
        random_state = None
        for i in range(seed):
            if random_state is None:
                random_state = np.random.RandomState(self.example_seen + self.cfg.SEED)
            else:
                random_state = np.random.RandomState(random_state.randint(0, int(1e5)))
        return random_state

    def store_cache(self):
        # for i, param in enumerate(self.net.parameters()):
        #    self.parameter_cache[i].copy_(param.data)
        self.cache = copy.deepcopy(self.net.state_dict())

    def load_cache(self):
        self.net.load_state_dict(self.cache)
        self.net.zero_grad()

    def get_loss_and_pseudo_update(self, x, y, task_ids):
        ret_dict_d = self.forward_net(x, y, task_ids)
        self.optimizer.zero_grad()
        ret_dict_d['loss'].backward(retain_graph=False)
        if isinstance(self.optimizer, torch.optim.SGD):
            step_wo_state_update_sgd(self.optimizer, amp=1.)
        elif isinstance(self.optimizer, torch.optim.Adam):
            step_wo_state_update_adam(self.optimizer, amp=1.)
        elif isinstance(self.optimizer, AdamW):
            step_wo_state_update_adamw(self.optimizer)
        else:
            raise NotImplementedError
        return ret_dict_d

    def decide_mir_mem(self, x, y, task_ids, mir_k, cand_x, cand_y, cand_task_ids, indices, mir_least):
        if cand_x.size(0) < mir_k:
            return cand_x, cand_y, cand_task_ids, indices
        else:
            self.store_cache()
            if type(cand_y[0]) not in [list, tuple]:
                cand_y = concat_with_padding(cand_y)
            else:
                cand_y = [torch.stack(_).to(x.device) for _ in zip(*cand_y)]
            with torch.no_grad():
                ret_dict_mem_before = self.forward_net(cand_x, cand_y, reduce=False, task_ids=cand_task_ids)
            ret_dict_d = self.get_loss_and_pseudo_update(x, y, task_ids)
            with torch.no_grad():
                ret_dict_mem_after = self.forward_net(cand_x, cand_y, reduce=False, task_ids=cand_task_ids)
                loss_increase = ret_dict_mem_after['loss'] - ret_dict_mem_before['loss']
            with torch.no_grad():
                if self.goal == 'captioning':
                    if self.mir_agg == 'avg':
                        loss_increase_by_ts = loss_increase.view(cand_x.size(0), -1).sum(1)
                        mask_num_by_ts = (cand_y[2] != -1).sum(1).float() + 1e-10
                        loss_increase = loss_increase_by_ts / mask_num_by_ts
                    elif self.mir_agg == 'max':
                        loss_increase, _ = loss_increase.view(cand_x.size(0), -1).max(1)

                _, topi = loss_increase.topk(mir_k, largest=not mir_least)
                if type(cand_y) is not list:
                    mem_x, mem_y, mem_task_ids = cand_x[topi], cand_y[topi], cand_task_ids[topi]
                else:
                    mem_x, mem_task_ids = cand_x[topi], cand_task_ids[topi]
                    mem_y = [_[topi] for _ in cand_y]

            self.load_cache()
            return mem_x, mem_y, mem_task_ids, indices[topi.cpu()]


    def sample_mem_batch(self, device, return_indices=False, k=None, seed=1,
                         mir=False, input_x=None, input_y=None, input_task_ids=None, mir_k=0,
                         skip_task=None, mir_least=False):
        random_state = self.get_random(seed)
        if k is None:
            k = self.mem_bs

        if not self.balanced:
            # reservoir
            n_max = min(self.mem_limit, self.example_seen)
            available_indices = [_ for _ in range(n_max)]
            if skip_task is not None and get_config_attr(self.cfg, 'EXTERNAL.REPLAY.FILTER_SELF', default=0, mute=True):
                available_indices = list(filter(lambda x: self.reservoir['y_extra'][x] != skip_task, available_indices))
            if not available_indices:
                if return_indices:
                    return None, None, None
                else:
                    return None, None, None
            elif len(available_indices) < k:
                indices = np.arange(n_max)
            else:
                indices = random_state.choice(available_indices, k, replace=False)
        else:
            available_index = self.get_available_index()
            if len(available_index) == 0:
                if return_indices:
                    return None, None, None
                else:
                    return None, None, None
            elif len(available_index) < k:
                indices = np.array(available_index)
            else:
                indices = random_state.choice(available_index, k, replace=False)
        x = self.reservoir['x'][indices]
        x = torch.from_numpy(x).to(device).float()

        y = index_select(self.reservoir['y'], indices, device)  # [  [...], [...] ]
        y_extra = index_select(self.reservoir['y_extra'], indices, device)
        if type(y[0]) not in [list, tuple]:
            y_pad = concat_with_padding(y)
        else:
            y_pad = [torch.stack(_).to(device) for _ in zip(*y)]
        y_extra = concat_with_padding(y_extra)

        if mir:
            x, y_pad, y_extra, indices = self.decide_mir_mem(input_x, input_y, input_task_ids, mir_k,
                                                             x, y, y_extra, indices, mir_least)

        if not return_indices:
            return x, y_pad, y_extra
        else:
            return (x, indices), y_pad, y_extra

    def observe(self, x, y, task_ids=None, extra=None, optimize=True):
        # recover image, feat from x
        if task_ids is None:
            task_ids = torch.zeros(x.size(0)).to(x.device).long()

        if self.mir:
            mem_x, mem_y, mem_task_ids = self.sample_mem_batch(x.device, input_x=x, input_y=y, input_task_ids=task_ids,
                                                               mir_k=self.mir_k, mir=self.mir,
                                                               skip_task=task_ids[0].item())
        else:
            mem_x, mem_y, mem_task_ids = self.sample_mem_batch(x.device, skip_task=task_ids[0].item())

        batch_size = x.size(0)
        if mem_x is not None and not self.separate_replay and not self.goal == 'captioning': # a dirty fix to prevent oom
            if not self.mem_augment:
                combined_x = torch.cat([x, mem_x], 0)
                combined_y = concat_with_padding([y, mem_y])
                combined_task_ids = concat_with_padding([task_ids, mem_task_ids])
            else:
                aug_mem_x = self.transform_image_batch(mem_x)
                combined_x = torch.cat([x,mem_x,aug_mem_x],0)
                combined_y = concat_with_padding([y,mem_y,mem_y])
                combined_task_ids = concat_with_padding([task_ids, mem_task_ids, mem_task_ids])
        else:
            combined_x, combined_y, combined_task_ids = x, y, task_ids

        ret_dict = self.forward_net(combined_x, combined_y, combined_task_ids,
                                    reduce=self.concat_replay or self.separate_replay)

        loss_tmp = ret_dict['loss']
        if optimize:
            # loss = loss_tmp.mean()
            # print(loss.item())
            if self.concat_replay or self.separate_replay:
                loss = ret_dict['loss']
            else:
                loss = loss_tmp[: x.size(0)].mean()
                if mem_x is not None:
                    loss += loss_tmp[x.size(0):].mean()

            self.optimizer.zero_grad()

            if self.concat_replay and mem_x is not None:
                loss = loss / 2

            loss.backward()

            #if mem_x is None or (not self.separate_replay and not self.goal == 'captioning'):
            if not self.concat_replay or mem_x is None:
                self.optimizer.step()

            if (self.separate_replay or self.goal == 'captioning') and mem_x is not None:
                ret_dict_mem = self.forward_net(mem_x, mem_y, mem_task_ids, reduce=True)

                if not self.concat_replay:
                    self.optimizer.zero_grad()

                if self.concat_replay:
                    ret_dict_mem['loss'] = ret_dict_mem['loss'] / 2

                ret_dict_mem['loss'].backward()
                self.optimizer.step()
                ret_dict['loss'] = (ret_dict['loss'] + ret_dict_mem['loss']) / 2

            for b in range(batch_size):  # x.size(0)
                if type(y) is tuple:
                    y_ = [_[b] for _ in y]
                else:
                    y_ = y[b]
                self.update_mem(x[b], y_, task_ids[b] if task_ids is not None else None,
                                x_feat=None)
        return ret_dict

    def dump_reservoir(self, path, verbose=False):
        f = open(path, 'wb')
        pickle.dump({
            'reservoir_x': self.reservoir['x'] if verbose else None,
            'reservoir_y': self.reservoir['y'],
            'reservoir_y_extra': self.reservoir['y_extra'],
            'mem_occupied': self.mem_occupied,
            'example_seen': self.example_seen,
            'seen_tasks': self.seen_tasks,
            'balanced': self.balanced
        }, f)
        f.close()

    def load_reservoir(self, path):
        try:
            f = open(path, 'rb')
            dic = pickle.load(f)
            for k in dic:
                setattr(self, k, dic[k])
            f.close()
            return dic
        except FileNotFoundError:
            print('no replay buffer dump file')
            return {}

    def load_reservoir_from_dic(self, dic):
        for k in dic:
            setattr(self, k, dic[k])

    def get_reservoir(self):
        return {'reservoir': self.reservoir, 'mem_occupied': self.mem_occupied,
                'example_seen': self.example_seen, 'seen_tasks': self.seen_tasks,
                'balanced': self.balanced}

    def mean_of_exemplar_classify(self, cropped_image_inp):
        if not hasattr(self, 'mean_exemplar_vec'):
            mean_exemplar_vec = []
            for task in self.seen_tasks:
                offset_start, offset_stop = self.compute_offset(task, len(self.seen_tasks))
                x = self.reservoir['x'][offset_start: min(offset_stop, offset_start + self.mem_occupied[task])]
                x = torch.from_numpy(x).float().to(cropped_image_inp.device)
                cropped_image = x[:, : 3 * self.cfg.EXTERNAL.IMAGE_SIZE * self.cfg.EXTERNAL.IMAGE_SIZE] \
                    .view(-1, 3, self.cfg.EXTERNAL.IMAGE_SIZE, self.cfg.EXTERNAL.IMAGE_SIZE)
                feat = self.net.get_obj_features(cropped_image)
                mean_feat = feat.mean(0)
                mean_exemplar_vec.append(mean_feat)
            self.mean_exemplar_vec = torch.stack(mean_exemplar_vec)  # [C, H]

        feat = self.net.get_obj_features(cropped_image_inp)  # [B, H]
        mean_exemplar_vec_expand = self.mean_exemplar_vec.unsqueeze(0).expand(feat.size(0), -1, -1)  # [B,C,H]
        feat_expand = feat.unsqueeze(1).expand(-1, self.mean_exemplar_vec.size(0), -1)  # [B,C,H]
        dist = torch.sum((mean_exemplar_vec_expand - feat_expand) ** 2, -1)  # [B,C]
        dist = torch.sqrt(dist)
        _, pred = dist.min(-1)  # [B]

        pred_index_fix = torch.zeros(pred.size()).to(pred.device)
        for b in range(pred.size(0)):
            pred_index_fix[b] = self.seen_tasks[pred[b]]

        return pred_index_fix


def step_wo_state_update_adam(adam, closure=None, amp=1.):
    """Performs a single optimization step. Do not update optimizer states
    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    if type(adam) is not Adam:
        raise ValueError
    loss = None
    if closure is not None:
        loss = closure()

    for group in adam.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad.data
            if grad.is_sparse:
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
            amsgrad = group['amsgrad']

            state = adam.state[p]

            # State initialization
            if len(state) == 0:
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data)

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            if amsgrad:
                max_exp_avg_sq = state['max_exp_avg_sq']
            beta1, beta2 = group['betas']

            # state['step'] += 1

            if group['weight_decay'] != 0:
                grad.add_(group['weight_decay'], p.data)

            # Decay the first and second moment running average coefficient
            exp_avg = exp_avg.mul(beta1).add(1 - beta1, grad)
            exp_avg_sq = exp_avg_sq.mul(beta2).addcmul(1 - beta2, grad, grad)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                # Use the max. for normalizing running avg. of gradient
                denom = max_exp_avg_sq.sqrt().add_(group['eps'])
            else:
                denom = exp_avg_sq.sqrt().add_(group['eps'])

            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']
            step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1 * amp

            p.data.addcdiv_(-step_size, exp_avg, denom)

    return loss


def step_wo_state_update_sgd(sgd, closure=None, amp=1.):
    """Performs a single optimization step.
    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    if closure is not None:
        loss = closure()

    for group in sgd.param_groups:
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']

        for p in group['params']:
            if p.grad is None:
                continue
            d_p = p.grad.data
            if weight_decay != 0:
                d_p.add_(weight_decay, p.data)
            if momentum != 0:
                param_state = sgd.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, d_p)
                if nesterov:
                    d_p = d_p.add(momentum, buf)
                else:
                    d_p = buf

            p.data.add_(-group['lr'] * amp, d_p)

    return loss


def step_wo_state_update_adamw(self, closure=None):
    """Performs a single optimization step.

    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    if closure is not None:
        loss = closure()

    for group in self.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad.data
            if grad.is_sparse:
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

            state = self.state[p]

            # State initialization
            #if len(state) == 0:
            #    state['step'] = 0
            #    # Exponential moving average of gradient values
            #    state['exp_avg'] = torch.zeros_like(p.data)
            #    # Exponential moving average of squared gradient values
            #    state['exp_avg_sq'] = torch.zeros_like(p.data)

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = group['betas']

            #state['step'] += 1

            # Decay the first and second moment running average coefficient
            # In-place operations to update the averages at the same time
            exp_avg = exp_avg.mul(beta1).add(1.0 - beta1, grad)
            exp_avg_sq = exp_avg_sq.mul(beta2).addcmul(1.0 - beta2, grad, grad)
            denom = exp_avg_sq.sqrt().add(group['eps'])

            step_size = group['lr']
            if group['correct_bias']:  # No bias correction for Bert
                bias_correction1 = 1.0 - beta1 ** state['step']
                bias_correction2 = 1.0 - beta2 ** state['step']
                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

            p.data.addcdiv_(-step_size, exp_avg, denom)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want to decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            # Add weight decay at the end (fixed version)
            if group['weight_decay'] > 0.0:
                p.data.add_(-group['lr'] * group['weight_decay'], p.data)

    return loss

def get_updated_weights_sgd(sgd, closure=None, amp=1.):
    """Performs a single optimization step.
    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    if closure is not None:
        loss = closure()
    weights = []
    for group in sgd.param_groups:
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']

        for p in group['params']:
            if p.grad is None:
                weights.append(p.data)
            d_p = p.grad
            if weight_decay != 0:
                d_p = d_p.add(weight_decay, p.data)
            if momentum != 0:
                param_state = sgd.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, d_p)
                if nesterov:
                    d_p = d_p.add(momentum, buf)
                else:
                    d_p = buf
            weights.append(p.data - group['lr'] * amp * d_p)
    return weights