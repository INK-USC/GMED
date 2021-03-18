from .initialization import get_glove_matrix
from torchvision.models import resnet50, resnet34 #resnet18
import torch
from torch import nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)
logger.setLevel('INFO')
logger.addHandler(logging.StreamHandler())


def get_resnet_features(x, net):
    if isinstance(net, ResNet):
        return net(x)
    else:
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.maxpool(x)

        x = net.layer1(x)
        x = net.layer2(x)
        x = net.layer3(x)
        x = net.layer4(x)

        x = net.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class ResNetClassifier(nn.Module):
    def __init__(self, cfg, depth='34', mlp=3, init=True, ignore_index=0, num_of_datasets=1,
                 num_of_classes=100, task_incremental=False, goal=None, *args, **kwargs):
        super().__init__()

        self.debug = hasattr(cfg, 'DEBUG') and cfg.DEBUG

        if depth == '34':
            self.resnet = resnet34(pretrained=False)
            resnet_feat_size = self.resnet.fc.weight.size(1)
        elif depth == '18':
            if goal == 'split_mini_imagenet':
                self.resnet = ResNet18(input_size=(3,84,84))
            else:
                self.resnet = ResNet18()
            resnet_feat_size = self.resnet.last_hid_size
        hidden_size = resnet_feat_size
        mlps = []
        for _ in range(mlp-1):
            mlps.append(nn.Linear(hidden_size, hidden_size))
            mlps.append(nn.ReLU())

        self.mlp_attr = nn.Sequential(
            *mlps
        )

        self.task_incremental = task_incremental
        self.num_of_datasets = num_of_datasets
        self.num_of_classes = num_of_classes

        if self.debug:
            self.final_layer = nn.Linear(self.resnet.last_hid_size, num_of_classes * num_of_datasets)
        else:
            self.final_layer = nn.Linear(hidden_size, num_of_classes * num_of_datasets)

        self.mlp_obj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 2000)
        )

        self.ignore_index = ignore_index
        self.criterion = F.cross_entropy

        self.cfg = cfg
        #self.task_specific_params = nn.ModuleList([self.mlp_attr[0], self.mlp_attr[2], self.mlp_attr[4]])

        if init and cfg.MODE == 'train':
            self.initialize()

    def forward(self, images, bbox_images, spatial_feat, attr_labels, obj_labels, weights=None, reduce=True,
                task_ids=None):
        if self.debug:
            feat = self.resnet.return_hidden(bbox_images)
            attr_score = self.final_layer(feat)
        else:
            feat = get_resnet_features(bbox_images, self.resnet)
            attr_feat = self.mlp_attr(torch.cat([feat], -1))
            attr_score = self.final_layer(attr_feat)

            if self.task_incremental:
                scores = [] # get logits of corresponding tasks
                for b in range(attr_score.size(0)):
                    scores.append(attr_score[b, task_ids[b] * self.num_of_classes: (task_ids[b] + 1) * self.num_of_classes])
                attr_score = torch.stack(scores)

        loss_attr, loss_obj = None, None
        if attr_labels is not None:
            reduction = 'mean' if reduce else 'none'
            loss_attr = self.criterion(attr_score, attr_labels, ignore_index=self.ignore_index,
                                       reduction=reduction)
        # if obj_labels is not None:
        #     reduction = 'mean' if reduce else 'none'
        #     loss_obj = self.criterion(obj_score, obj_labels, ignore_index=self.ignore_index,
        #                               reduction=reduction)

        return {
            'attr_score': attr_score,
            #'obj_score': obj_score,
            'attr_loss': loss_attr,
            #'obj_loss': loss_obj,
            'loss': loss_attr, #loss_obj + loss_attr,
            'score': attr_score,
            'feat': feat
        }

    def forward_from_feat(self, feat, attr_labels, reduce=True, **kwargs):
        reduction = 'mean' if reduce else 'none'
        attr_score = self.mlp_attr(feat)
        loss_attr = self.criterion(attr_score, attr_labels, ignore_index=self.ignore_index,
                                   reduction=reduction)
        return {
            'attr_score': attr_score,
            'attr_loss': loss_attr,
            'loss': loss_attr,
            'score': attr_score,
            'feat': feat
        }

    def get_obj_features(self, bbox_images):
        feat = get_resnet_features(bbox_images, self.resnet)
        for i, module in enumerate(self.mlp_obj._modules.values()):
            if i == len(self.mlp_obj) - 1: break
            feat = module(feat)
        return feat

    def set_task_specific_weights(self, idx, weight):
        self.task_specific_params[idx].weight = weight

    def _freeze_resnet_if_required(self, cfg):
        if hasattr(cfg.EXTERNAL, 'FREEZE_RESNET') and cfg.EXTERNAL.FREEZE_RESNET:
            for param in self.resnet.parameters():
                param.requires_grad = False
            logger.info('Resnet frozen')

    def _load_resnet_weights_if_required(self, cfg):
        if hasattr(cfg.EXTERNAL, 'LOAD_RESNET') and cfg.EXTERNAL.LOAD_RESNET:
            checkpoint = torch.load(cfg.EXTERNAL.LOAD_RESNET, map_location=torch.device("cpu"))
            model_state_dict = checkpoint['model']
            model_state_dict = dict(filter(lambda x: x[0].startswith('resnet'), model_state_dict.items()))
            model = ResNetClassifier(self.cfg, init=False)
            model.load_state_dict(model_state_dict, strict=False)
            self.resnet = model.resnet.to(cfg.MODEL.DEVICE)
            logger.info('loaded resnet from %s' % cfg.EXTERNAL.LOAD_RESNET)

    def initialize(self):
        self._freeze_resnet_if_required(self.cfg)
        self._load_resnet_weights_if_required(self.cfg)


class ResNetClassifierWObj(ResNetClassifier):
    def __init__(self, cfg, init=True, *args, **kwargs):
        # do not init
        super().__init__(cfg, init=False, *args, **kwargs)
        self.cfg = cfg
        self.resnet = resnet34(pretrained=True)
        resnet_feat_size = self.resnet.fc.weight.size(1)
        hidden_size = resnet_feat_size
        label_embed_size = self.cfg.WOBJ.LABEL_EMBED_DIM

        self.obj_emb = nn.Embedding(2000, label_embed_size)
        self.mlp_attr = nn.Sequential(
            nn.Linear(hidden_size + label_embed_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 101)
        )
        self.task_specific_params = nn.ModuleList([self.mlp_attr[0], self.mlp_attr[2], self.mlp_attr[4]])
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        if init and cfg.MODE == 'train':
            self.initialize()

    def forward(self, images, bbox_images, spatial_feat, attr_labels, obj_labels, weights=None):
        obj_emb = self.obj_emb(obj_labels)
        feat = get_resnet_features(bbox_images, self.resnet)
        feat = torch.cat([feat, obj_emb], -1)

        if weights is None:
            attr_score = self.mlp_attr(feat)
        else: # hypernetwork
            hidden_1 = F.relu(F.linear(feat, weights[0], weights[1]))
            hidden_2 = F.relu(F.linear(hidden_1, weights[2], weights[3]))
            attr_score = F.linear(hidden_2, weights[4], weights[5])

        loss_attr, loss_obj = None, None
        if attr_labels is not None:
            loss_attr = self.criterion(attr_score, attr_labels)
        #if obj_labels is not None:
        #    loss_obj = self.criterion(obj_score, obj_labels)

        return {
            'attr_score': attr_score,
            #'obj_score': obj_score,
            'attr_loss': loss_attr,
            #'obj_loss': loss_obj,
            'loss': loss_attr
        }

    def get_obj_features(self, bbox_images):
        feat = get_resnet_features(bbox_images, self.resnet)
        for i, module in enumerate(self.mlp_obj._modules.values()):
            if i == len(self.mlp_obj) - 1: break
            feat = module(feat)
        return feat

    def set_task_specific_weights(self, idx, weight):
        self.task_specific_params[idx].weight = weight

    def initialize_word_emb(self, vocab, w2v_file):
        device = self.obj_emb.weight.data.device
        mat = get_glove_matrix(vocab, w2v_file, self.obj_emb.weight.data.cpu().numpy())
        mat = torch.from_numpy(mat).float().to(device)
        self.obj_emb.weight.data = mat
        logger.info('loaded embedding from {}'.format(w2v_file))

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, nf, input_size):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.input_size = input_size

        self.conv1 = conv3x3(input_size[0], nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        #self.bn1  = CategoricalConditionalBatchNorm(nf, 2)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        # hardcoded for now
        self.last_hid_size = nf * 8 * block.expansion if input_size[1] in [8,16,21,32,42] else 640
        #self.linear = nn.Linear(last_hid, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def return_hidden(self, x):
        bsz = x.size(0)
        #pre_bn = self.conv1(x.view(bsz, 3, 32, 32))
        #post_bn = self.bn1(pre_bn, 1 if is_real else 0)
        #out = F.relu(post_bn)
        out = F.relu(self.bn1(self.conv1(x.view(bsz, *self.input_size))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        out = self.return_hidden(x)
        #out = self.linear(out)
        return out

def ResNet18(nf=20, input_size=(3, 32, 32)):
    return ResNet(BasicBlock, [2, 2, 2, 2], nf, input_size)