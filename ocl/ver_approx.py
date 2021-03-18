from .ver import *
from torch.nn import functional as F
import random


class ExperienceEvolveApprox(FOExperienceEvolve):
    def __init__(self, base, optimizer, input_size, cfg, goal):
        super().__init__(base, optimizer, input_size, cfg, goal)
        self.edit_least = get_config_attr(cfg, 'EXTERNAL.OCL.EDIT_LEAST', default=0)
        self.edit_random = get_config_attr(cfg, 'EXTERNAL.OCL.EDIT_RANDOM', default=0)

        self.edit_interfere = get_config_attr(cfg, 'EXTERNAL.OCL.EDIT_INTERFERE', default=1)
        self.edit_replace = get_config_attr(cfg, 'EXTERNAL.OCL.EDIT_REPLACE', default=0)
        self.replace_reweight = get_config_attr(cfg, 'EXTERNAL.OCL.REPLACE_REWEIGHT', default=0)
        self.use_relu = get_config_attr(cfg, 'EXTERNAL.OCL.USE_RELU', default=0)
        self.reg_strength = get_config_attr(cfg, 'EXTERNAL.OCL.REG_STRENGTH', default=0.1)
        self.always_proj = get_config_attr(cfg, 'EXTERNAL.OCL.ALWAYS_PROJ', default=0)

        self.edit_mir_k = get_config_attr(cfg, 'EXTERNAL.OCL.EDIT_K', default=-1)

        self.hal_mem = get_config_attr(cfg, 'EXTERNAL.OCL.HAL_MEM', default=0)
        self.post_edit_mem_aug = get_config_attr(cfg,'EXTERNAL.OCL.POST_EDIT_MEM_AUG', default=1)
        self.edit_aug_mem = get_config_attr(cfg,'EXTERNAL.OCL.EDIT_AUG_MEM', default=0)
        self.double_weight = get_config_attr(cfg,'EXTERNAL.OCL.DOUBLE_WEIGHT', default=0)

        if self.edit_mir_k == -1:
            self.edit_mir_k = self.mir_k

    def sample_mem_batch_same_task(self, device, task_id_or_label, return_indices=False, mem_k=None, seed=0, use_same_label=False):
        if mem_k is None:
            mem_k = self.mem_bs
        if use_same_label:
            label = task_id_or_label
        else:
            task_id = task_id_or_label

        n_max = min(self.mem_limit, self.example_seen)
        indices = []
        for i in range(n_max):
            if use_same_label:
                if self.reservoir['y'][i] == label:
                    indices.append(i)
            else:
                if self.reservoir['y_extra'][i] == task_id:
                    indices.append(i)

        # reservoir
        if not indices:
            return None, None, None
        elif len(indices) >= mem_k:
            indices = np.random.RandomState(seed * self.example_seen + self.cfg.SEED).\
                choice(indices, mem_k, replace=False)

        x = self.reservoir['x'][indices]
        #x_origin = self.reservoir['x_origin'][indices]

        x = torch.from_numpy(x).to(device).float()
        #x_origin = torch.from_numpy(x_origin).to(device).float()
        y = index_select(self.reservoir['y'], indices, device) # [  [...], [...] ]
        y_extra = index_select(self.reservoir['y_extra'], indices, device)
        y_extra = concat_with_padding(y_extra)
        if type(y[0]) not in [list, tuple]:
            y_pad = concat_with_padding(y)
        else:
            y_pad = [torch.stack(_).to(device) for _ in zip(*y)]

        if not return_indices:
            return x, y_pad, y_extra
        else:
            return (x, indices), y_pad, y_extra

    def clear_mem_grad(self, mem_x):
        mem_x.detach_()
        mem_x.grad = None
        mem_x.requires_grad = True


    def observe(self, x, y, task_ids, extra=None, optimize=True, sequential=False):
        n_iter = get_config_attr(self.cfg, 'EXTERNAL.OCL.N_ITER', default=1, mute=True)
        batch_size = x.size(0)
        self.store_cache()
        for i_iter in range(n_iter):
            if not self.mir:
                mem_x_indices, mem_y, mem_task_ids = self.sample_mem_batch(x.device, return_indices=True, seed=i_iter + 1)
                if self.edit_random: # select another batch for editing
                    edit_x_indices, edit_y, edit_task_ids = self.sample_mem_batch(x.device, return_indices=True,
                                                                               k=self.mir_k, seed=i_iter + 2)
                else:
                    edit_x_indices, edit_y, edit_task_ids = mem_x_indices, mem_y, mem_task_ids
            else:
                mem_x_indices, mem_y, mem_task_ids = self.sample_mem_batch(x.device, return_indices=True,
                                                                           input_x=x, input_y=y, input_task_ids=task_ids,
                                                                           mir_k=self.mir_k, mir=self.mir,
                                                                           skip_task=task_ids[0].item(),
                                                                           seed=i_iter + 1
                                                                           )
                if self.edit_least:
                    edit_x_indices, edit_y, edit_task_ids = self.sample_mem_batch(x.device, return_indices=True,
                                                                               input_x=x, input_y=y, input_task_ids=task_ids,
                                                                               mir_k=self.edit_mir_k, mir=self.mir,
                                                                               skip_task=task_ids[0].item(),
                                                                               mir_least=True,
                                                                               seed=i_iter + 2
                                                                               )
                elif self.edit_random:
                    edit_x_indices, edit_y, edit_task_ids = self.sample_mem_batch(x.device, return_indices=True,
                                                                               k=self.edit_mir_k, seed=i_iter + 2)
                else:
                    edit_x_indices, edit_y, edit_task_ids = mem_x_indices, mem_y, mem_task_ids
            self.optimizer.zero_grad()

            edit_x_val_indices, edit_y_val, _ = self.sample_mem_batch_same_task(x.device, task_ids.cpu().numpy()[0],
                                                                              return_indices=True, seed=i_iter + 2,
                                                                              mem_k=self.mir_k if self.mir else self.mem_bs)
            edit_task_ids_val = task_ids
            #mem_x_val_indices, _, mem_y_val, mem_task_ids_val = self.sample_mem_batch(x.device, return_indices=True, seed=1)

            if edit_x_indices is None:
                combined_x, combined_y, combined_task_ids = x, y, task_ids
            else:
                mem_x, mem_indices = mem_x_indices
                edit_x, edit_indices = edit_x_indices
                if edit_x_val_indices is not None:
                    edit_x_val, indices_val = edit_x_val_indices
                else:
                    edit_x_val, indices_val = None, None
                if self.mem_augment:
                    aug_mem_x = self.transform_image_batch(mem_x)
                if i_iter == 0 and self.edit_interfere:
                    if self.hal_mem:
                        train_x, train_y, _ = self.sample_mem_batch(x.device, return_indices=False,
                                                                    seed=i_iter + 3)
                    else:
                        train_x, train_y = x, y

                    edit_x, mem_x = self.edit_mem_interfere(train_x, train_y, task_ids, mem_x, mem_y, edit_x, edit_y, edit_task_ids,
                           edit_x_val, edit_y_val, edit_task_ids_val, edit_indices)
                    if not self.no_write_back:
                        self.evolve_mem(edit_x, edit_indices)

                    # load cached parameters back
                    self.load_cache()
                if not self.mem_augment:
                    combined_x = torch.cat([x, mem_x], 0)
                    combined_y = concat_with_padding([y, mem_y])
                    combined_task_ids = concat_with_padding([task_ids, mem_task_ids])
                else:
                    if self.post_edit_mem_aug:
                        aug_mem_x = self.transform_image_batch(mem_x)
                    if self.edit_aug_mem:
                        aug_mem_x, _ = self.edit_mem_interfere(train_x, train_y, task_ids, aug_mem_x, mem_y, aug_mem_x,
                                                                mem_y, edit_task_ids,
                                                                edit_x_val, edit_y_val, edit_task_ids_val, edit_indices)
                    if self.double_weight:
                        combined_x = torch.cat([x, mem_x, aug_mem_x], 0)
                        combined_y = concat_with_padding([y, mem_y, mem_y])
                        combined_task_ids = concat_with_padding([task_ids, mem_task_ids, mem_task_ids])
                    else:
                        combined_x = torch.cat([x, mem_x, aug_mem_x], 0)
                        combined_y = concat_with_padding([y, mem_y, mem_y])
                        combined_task_ids = concat_with_padding([task_ids, mem_task_ids, mem_task_ids])

            ret_dict = self.forward_net(combined_x, combined_y, task_ids=combined_task_ids, reduce=False)
            loss_tmp = ret_dict['loss']
            if optimize:
                loss = loss_tmp[: x.size(0)].mean()
                if mem_x_indices is not None:
                    loss += loss_tmp[x.size(0):].mean() #* (2 if self.double_weight else 1)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        if self.lb_reservoir and mem_x_indices is not None:
            self.update_loss_states(loss_tmp[x.size(0):], mem_indices)
        for b in range(batch_size):
            if type(y) is tuple:
                self.update_mem(x[b], [_[b] for _ in y], task_ids[b])
            else:
                self.update_mem(x[b], y[b], task_ids[b])

        return ret_dict

    def edit_mem_interfere(self, x, y, task_ids, mem_x, mem_y, edit_x, edit_y, edit_task_ids,
                           edit_x_val, edit_y_val, edit_task_ids_val, edit_indices):
        """
        Edit memory so that they are more inter
        :param x:
        :param y:
        :param task_ids:
        :param mem_x:
        :param mem_y:
        :param edit_x:
        :param edit_y:
        :param edit_task_ids:
        :param edit_x_val:
        :param edit_y_val:
        :param edit_task_ids_val:
        :return:
        """
        device = x.device
        # only edit at the first iter

        for i in range(self.grad_iter):
            # evaluate loss on edit_x, edit_y
            self.clear_mem_grad(edit_x)
            # evaluate grad of l wrt edit
            ret_dict_edit_before = self.forward_net(edit_x, edit_y, reduce=True, task_ids=edit_task_ids)
            # train the model on D
            grad_reg = -torch.autograd.grad(torch.sum(ret_dict_edit_before['loss']),
                                            edit_x, retain_graph=True)[0]
            ret_dict_edit_before['loss'].backward()
            edit_x_grad1 = edit_x.grad

            self.clear_mem_grad(edit_x)

            for _ in range(1):
                ret_dict_d = self.forward_net(x, y, task_ids=task_ids)
                self.optimizer.zero_grad()
                ret_dict_d['loss'].backward(retain_graph=False)
                if isinstance(self.optimizer, torch.optim.SGD):
                    step_wo_state_update_sgd(self.optimizer, amp=1.)
                elif isinstance(self.optimizer, torch.optim.Adam):
                    step_wo_state_update_adam(self.optimizer, amp=1.)
                else:
                    raise NotImplementedError

            ret_dict_edit_after = self.forward_net(edit_x, edit_y, reduce=True, task_ids=edit_task_ids)
            if 'mask_cnts' not in ret_dict_edit_after:
                loss_increase = ret_dict_edit_after['loss'] - ret_dict_edit_before['loss']
            else:
                loss_increase = (ret_dict_edit_after['loss'] - ret_dict_edit_before['loss']).sum() / \
                                (sum(ret_dict_edit_after['mask_cnts']) + 1e-10)

            #if self.use_relu:
            #    loss_increase = F.relu(loss_increase)
            ret_dict_edit_after['loss'].backward()
            edit_x_grad2 = edit_x.grad
            grad_delta = edit_x_grad2 - edit_x_grad1

            grad_delta_2 = 0

            self.clear_mem_grad(edit_x)
            self.load_cache()

            total_grad = 0

            if self.cfg.EXTERNAL.OCL.USE_LOSS_1:
                if self.cfg.EXTERNAL.OCL.USE_LOSS_1 == 1:
                    total_grad += self.cfg.EXTERNAL.OCL.USE_LOSS_1 * grad_delta
                elif self.cfg.EXTERNAL.OCL.USE_LOSS_1 == -2: # random tied direction uniform norm
                    random_vecs = self.get_random_grad(grad_delta, edit_indices)
                    total_grad += random_vecs
                elif self.cfg.EXTERNAL.OCL.USE_LOSS_1 == -3: # random tied direction keep norm
                    random_vecs = self.get_direction_perturbed_grad(grad_delta, edit_indices)
                    total_grad += random_vecs
                elif self.cfg.EXTERNAL.OCL.USE_LOSS_1 == -4: # random untied direction keep norm
                    random_vecs = self.get_direction_perturbed_grad_untied(grad_delta, edit_indices)
                    total_grad += random_vecs
                elif self.cfg.EXTERNAL.OCL.USE_LOSS_1 == -5: # negated gradient
                    neg_vecs = self.get_neg_gradients(grad_delta, edit_indices)
                    total_grad += neg_vecs
                elif self.cfg.EXTERNAL.OCL.USE_LOSS_1 == -6: # adversarial-continuous
                    total_grad += edit_x_grad1
                elif self.cfg.EXTERNAL.OCL.USE_LOSS_1 == -7: # untied, fixed
                    random_vecs = self.get_direction_perturbed_grad_untied_fixed(grad_delta, edit_indices)
                    total_grad += random_vecs
                elif self.cfg.EXTERNAL.OCL.USE_LOSS_1 == -8: # PGD
                    total_grad += torch.sign(edit_x_grad1)
                else:
                    raise ValueError

            if self.cfg.EXTERNAL.OCL.USE_LOSS_2:
                total_grad += self.cfg.EXTERNAL.OCL.USE_LOSS_2 * grad_delta_2

            if type(total_grad) is not int:  # has grad update
                #if self.cfg.EXTERNAL.OCL.PROJ_LOSS_REG:
                if self.cfg.EXTERNAL.OCL.PROJ_LOSS_REG == 1:
                    for b in range(total_grad.size(0)):
                        total_grad[b] = proj_grad(-grad_reg[b], total_grad[b], binary=False, always_proj=self.always_proj)
                elif self.cfg.EXTERNAL.OCL.PROJ_LOSS_REG == 2:
                    for b in range(total_grad.size(0)):
                        total_grad[b] = -grad_reg[b] * self.reg_strength + total_grad[b]

                mem_ages = self.get_mem_ages(edit_indices, astype=edit_x)
                stride_decayed = (1 - self.edit_decay) ** mem_ages

                for b in range(total_grad.size(0)):
                    edit_x[b] = edit_x[b] + self.grad_stride * stride_decayed[b] * total_grad[b]
            edit_x = edit_x.detach()
            mem_x = mem_x.detach()

        return edit_x, mem_x

    # code 2: edit memory to a random direction tied to each input example
    def get_random_grad(self, grad, indices):
        if not hasattr(self, 'random_dirs'):
            self.random_dirs = torch.zeros(self.mem_limit, *grad[0].size()).uniform_(-1,1)
        random_vecs = []
        for i, indice in enumerate(indices):
            random_vec = self.random_dirs[indice].to(grad.device)
            random_vec = random_vec / random_vec.norm()
            random_vecs.append(random_vec)
        random_vecs = torch.stack(random_vecs)
        return random_vecs

    # code 3: perturbate editing direction - keep norm
    def get_direction_perturbed_grad(self, grad, indices):
        if not hasattr(self, 'random_dirs'):
            self.random_dirs = torch.zeros(self.mem_limit, *grad[0].size()).uniform_(-1, 1)
        random_vecs = []
        for i, indice in enumerate(indices):
            random_vec = self.random_dirs[indice].to(grad.device)
            random_vec = random_vec / random_vec.norm() * grad[i].norm()
            random_vecs.append(random_vec)
        random_vecs = torch.stack(random_vecs)
        return random_vecs

    # code 4: perturbate editing direction - not tied
    def get_direction_perturbed_grad_untied(self, grad, indices):
        self.random_dirs = torch.zeros(self.mem_limit, *grad[0].size()).to(grad.device).uniform_(-1, 1)
        random_vecs = []
        for i, indice in enumerate(indices):
            random_vec = self.random_dirs[indice]
            random_vec = random_vec / random_vec.norm() * grad[i].norm()
            random_vecs.append(random_vec)
        random_vecs = torch.stack(random_vecs)
        return random_vecs

    # code 5: flip the update direction
    def get_neg_gradients(self, grad, indices):
        return -grad

    # code 7: untied and fixed
    def get_direction_perturbed_grad_untied_fixed(self, grad, indices):
        self.random_dirs = torch.zeros(len(indices), *grad[0].size()).to(grad.device).normal_(0, 1)
        random_vecs = []
        for i, indice in enumerate(indices):
            random_vec = self.random_dirs[i]
            random_vec = random_vec / random_vec.norm()
            random_vecs.append(random_vec)
        random_vecs = torch.stack(random_vecs)
        return random_vecs

    def to_mem_type(self, x, y, y_extra):
        x = x.cpu().numpy()
        if type(y) not in [list, tuple]:
            y = y_to_np(y)
        else:
            y = y_to_cpu(y)
        if type(y_extra) not in [list, tuple]:
            y_extra = y_to_np(y_extra)
        else:
            y_extra = y_to_cpu(y_extra)
        return x, y, y_extra

    def indices_to_examples(self, indices, device):
        cand_x = self.reservoir['x'][indices]
        cand_x = torch.from_numpy(cand_x).to(device).float()
        cand_y = index_select(self.reservoir['y'], indices, device)  # [  [...], [...] ]
        cand_y_extra = index_select(self.reservoir['y_extra'], indices, device)
        if type(cand_y[0]) not in [list, tuple]:
            cand_y_pad = concat_with_padding(cand_y)
        else:
            cand_y_pad = [torch.stack(_).to(device) for _ in zip(*cand_y)]
        cand_y_extra = concat_with_padding(cand_y_extra)
        return cand_x, cand_y_pad, cand_y_extra