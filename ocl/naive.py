from torch import nn
from torch import optim
import torch
from utils.utils import get_config_attr
import numpy as np

class NaiveWrapper(nn.Module):
    def __init__(self, base, optimizer, input_size, cfg, goal, **kwargs):
        super().__init__()
        self.net = base
        self.optimizer = optimizer
        self.input_size = input_size
        self.cfg = cfg
        self.goal = goal

        if 'caption' in self.goal:
            self.clip_grad = True
            self.use_image_feat = get_config_attr(self.cfg, 'EXTERNAL.USE_IMAGE_FEAT', default=0)
            self.spatial_feat_shape = (2048, 7, 7)
            self.bbox_feat_shape = (100, 2048)
            self.bbox_shape = (100,4)
        self.rev_optim = None
        if hasattr(self.net, 'rev_update_modules'):
            self.rev_optim = optim.Adam(lr=cfg.SOLVER.BASE_LR, betas=(0.9, 0.999),
                                        params=self.net.rev_update_modules.parameters())

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def forward_net(self, x, y, task_ids=None, **kwargs):
        if self.goal in ['classification']:
            if type(y) is tuple:
                attr_labels, obj_labels = y
            else:
                attr_labels, obj_labels = y, None
            cropped_image = x[:, : 3 * self.cfg.EXTERNAL.IMAGE_SIZE * self.cfg.EXTERNAL.IMAGE_SIZE] \
                .view(-1, 3, self.cfg.EXTERNAL.IMAGE_SIZE, self.cfg.EXTERNAL.IMAGE_SIZE)
            images = x[:, 3 * self.cfg.EXTERNAL.IMAGE_SIZE * self.cfg.EXTERNAL.IMAGE_SIZE:] \
                .view(-1, 3, self.cfg.EXTERNAL.IMAGE_SIZE, self.cfg.EXTERNAL.IMAGE_SIZE)
            ret_dict = self.net(
                bbox_images=cropped_image, spatial_feat=None,
                images=images,
                attr_labels=attr_labels,
                obj_labels=obj_labels,
                task_ids=task_ids
                **kwargs
             )
        elif self.goal == 'captioning':
            # legacy
            if self.use_image_feat:
                images = x.view(-1, 3, self.cfg.EXTERNAL.IMAGE_SIZE, self.cfg.EXTERNAL.IMAGE_SIZE)
                captions, caption_lens, labels = y
                ret_dict = self.net(images=images, captions=captions, caption_lens=caption_lens, labels=labels, **kwargs)
            else:
                # rebuild spatial and packed inputs
                batch_size = x.size(0)
                bfeat_dim = np.prod(self.bbox_feat_shape)
                # bbox_feats, spatial_feats, bboxes = x[:, :bfeat_dim].view(batch_size, *self.bbox_feat_shape), \
                #                                     x[:,bfeat_dim:bfeat_dim + sfeat_dim].view(batch_size, *self.spatial_feat_shape), \
                #                                     x[:,bfeat_dim + sfeat_dim:].view(batch_size, *self.bbox_shape)
                bbox_feats, bboxes = x[:, :bfeat_dim].view(batch_size, *self.bbox_feat_shape), \
                                     x[:,bfeat_dim:].view(batch_size, *self.bbox_shape)
                captions, caption_lens, labels = y
                ret_dict = self.net(bbox_feats=bbox_feats, captions=captions, caption_lens=caption_lens, labels=labels,
                                    bboxes=bboxes, **kwargs)

        elif self.goal == 'split_mnist' or self.goal == 'permute_mnist' or self.goal == 'rotated_mnist':
            images = x.view(x.size(0), -1)
            ret_dict = self.net(images, y, task_ids=task_ids, **kwargs)
        elif self.goal == 'split_cifar':
            images = x.view(-1, 3, 32, 32)
            ret_dict = self.net(bbox_images=images, spatial_feat=None, images=None, attr_labels=y, obj_labels=None,
                                task_ids=task_ids, **kwargs)
        elif self.goal == 'split_mini_imagenet':
            images = x.view(-1,3,84,84)
            ret_dict = self.net(bbox_images=images, spatial_feat=None, images=None, attr_labels=y, obj_labels=None,
                                task_ids=task_ids, **kwargs)
        else:
            raise ValueError
        return ret_dict

    def observe(self, x, y, task_ids, optimize=True):
        # if deprecated is not None:
        #     y = (y, deprecated)
        # recover image, feat from x
        self.optimizer.zero_grad()
        ret_dict = \
            self.forward_net(x, y, task_ids)

        loss = ret_dict['loss']
        if optimize:
            loss.backward()
            #if self.clip_grad:
            #    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()

        return ret_dict

    def initialize_word_emb(self, *args, **kwargs):
        self.net.initialize_word_emb(*args, **kwargs)
