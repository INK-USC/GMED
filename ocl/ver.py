from .er import *
import math
from torch.optim import Adam
import copy
from utils.utils import get_config_attr

class FOExperienceEvolve(ExperienceReplay):
    def __init__(self, base, optimizer, input_size, cfg, goal):
        super().__init__(base, optimizer, input_size, cfg, goal)
        # self.reservoir = {'x': np.zeros((self.mem_limit, input_size)),
        #                   'y': [None] * self.mem_limit,
        #                   'y_extra': [None] * self.mem_limit,
        #                   'x_origin': np.zeros((self.mem_limit, input_size)),
        #                   'x_edit_state': [None] * self.mem_limit,
        #                   'loss_stats': [None] * self.mem_limit,
        #                   'loss_stat_steps': [None] * self.mem_limit,
        #                   'forget': [None] * self.mem_limit,
        #                   'support': [None] * self.mem_limit
        #                   }
        self.itf_cnt = 0
        self.total_cnt = 0
        self.grad_iter = get_config_attr(cfg, 'EXTERNAL.OCL.GRAD_ITER', default=1)
        self.grad_stride = get_config_attr(cfg, 'EXTERNAL.OCL.GRAD_STRIDE', default=10.)
        self.edit_decay = get_config_attr(cfg, 'EXTERNAL.OCL.EDIT_DECAY', default=0.)
        self.no_write_back = get_config_attr(cfg, 'EXTERNAL.OCL.NO_WRITE_BACK', default=0)
        self.reservoir['age'] = np.zeros(self.mem_limit)

    def get_mem_ages(self, indices, astype):
        ages = self.reservoir['age'][indices]
        if torch.is_tensor(astype):
            ages = torch.from_numpy(ages).float().to(astype.device)
        return ages

    def observe(self, x, y, task_ids, extra=None, optimize=True, sequential=False):
        sequential = True
        global total_cnt, itf_cnt
        self.optimizer.zero_grad()
        mem_x_indices, mem_x_origin, mem_y, mem_task_ids = self.sample_mem_batch(x.device, return_indices=True)

        batch_size = x.size(0)
        if mem_x_indices is None:
            combined_x, combined_y, combined_task_ids = self.sample_mem_batch()
        else:
            mem_x, indices = mem_x_indices
            self.store_cache()
            for i in range(self.grad_iter):
                # evaluate loss on mem_x, mem_y
                mem_x.requires_grad = True
                mem_x.grad = None
                mem_x_origin.requires_grad = True

                # evaluate grad of l wrt mem
                self.optimizer.zero_grad()
                ret_dict_mem_before = self.forward_net(mem_x, mem_y, reduce=False, task_ids=task_ids)
                # grad_l = -torch.autograd.grad(torch.sum(ret_dict_mem_origin_before['loss']), mem_x_origin, retain_graph=True)[0]

                # train the model on D
                if not sequential:
                    self.get_loss_and_pseudo_update(x, y, task_ids)
                else:
                    self.train(False)
                    for b in range(batch_size):
                        x_b = x[b].unsqueeze(0)
                        if type(y) in [tuple, list]:
                            y_b = [_[b].unsqueeze(0) for _ in y]
                        else:
                            y_b = y[b].unsqueeze(0)
                        ret_dict_db = self.forward_net(x_b, y_b)
                        self.optimizer.zero_grad()
                        ret_dict_db['loss'].backward()
                        if isinstance(self.optimizer, torch.optim.SGD):
                            step_wo_state_update_sgd(self.optimizer, amp=1.)
                        elif isinstance(self.optimizer, torch.optim.Adam):
                            step_wo_state_update_adam(self.optimizer, amp=1.)
                        else:
                            raise NotImplementedError
                    self.train(True)

                ret_dict_mem_after = self.forward_net(mem_x, mem_y, reduce=False)
                if 'mask_cnts' not in ret_dict_mem_after:
                    loss_increase = (ret_dict_mem_after['loss'] - ret_dict_mem_before['loss']).mean()
                else:
                    loss_increase = (ret_dict_mem_after['loss'] - ret_dict_mem_before['loss']).sum() / \
                                        (sum(ret_dict_mem_after['mask_cnts']) + 1e-10)
                loss_increase.backward(retain_graph=False)
                grad_delta = mem_x.grad

                self.load_cache()

                mem_ages = self.get_mem_ages(mem_x_indices, astype=mem_x)
                stride_decayed = (1 - self.edit_decay) ** mem_ages

                proposed_mem_x = mem_x + self.grad_stride * stride_decayed.view(-1,1) * grad_delta
                proposed_mem_x.detach_()

                mem_x = proposed_mem_x
                mem_x = mem_x.detach()
            self.evolve_mem(mem_x, indices)

            # load cached parameters back
            self.load_cache()
            combined_x = torch.cat([x, mem_x], 0)
            combined_y = concat_with_padding([y, mem_y])

        ret_dict = self.forward_net(combined_x, combined_y)

        for b in range(batch_size):
            if type(y) is tuple:
                self.update_mem(x[b], [_[b] for _ in y], extra[b] if extra is not None else None)
            else:
                self.update_mem(x[b], y[b], extra[b] if extra is not None else None)

        loss = ret_dict['loss']
        if optimize:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return ret_dict

    def evolve_mem(self, x, indices):
        for i, idx in enumerate(indices):
            #if self.reservoir['age'][idx] < 10:
            self.reservoir['x'][idx] = x[i].cpu().numpy()
            self.reservoir['age'][idx] += 1



total_cnt, itf_cnt = 0, 0
def proj_grad(a, b, binary, always_proj):
    # project b to the direction of a

    dotp = torch.dot(a,b)
    if dotp >= 0:
        if not always_proj:
            return b
        else:
            return b - (torch.dot(-a,b) / torch.dot(a,a)) * a
    else:
        if binary: return torch.zeros_like(a)
        else: return b - (torch.dot(a,b) / torch.dot(a,a)) * a
