import torch.nn.functional as func
import torch.nn as nn
import torch

class GatedDense(nn.Module):
    def __init__(self, input_size, output_size, activation=None):
        super(GatedDense, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.h = nn.Linear(input_size, output_size)
        self.g = nn.Linear(input_size, output_size)

    def forward(self, x):
        h = self.h(x)
        if self.activation is not None:
            h = self.activation( self.h( x ) )

        g = self.sigmoid( self.g( x ) )

        return h * g

class FC2Layers(nn.Module):
    def __init__(self, **kwargs):
        super(FC2Layers, self).__init__()
        layer1_width = 400

        self.ds_idx = 0
        self.num_of_datasets = kwargs.get("num_of_datasets", 1)
        self.num_of_classes = kwargs.get("num_of_classes", 10)
        self.input_size = kwargs.get("input_size", 784)
        self.task_incremental = kwargs.get('task_incremental', False)

        act = nn.ReLU()
        self.layer1 = nn.Sequential(
            nn.Linear(self.input_size, layer1_width),
            nn.ReLU(),
            nn.Linear(layer1_width, layer1_width),
            nn.ReLU()
        )

        self.last_layer = nn.Linear(layer1_width, self.num_of_classes * self.num_of_datasets)
        self.criterion = func.cross_entropy


    def forward(self, x, y, task_ids=None, reduce=True, from_weights=False, weights=None):
        x = x.view(-1, self.input_size)
        if from_weights:
            out = func.relu(func.linear(x, weights[0], weights[1]))
            feat = func.relu(func.linear(out, weights[2], weights[3]))
            out = func.linear(feat, weights[4], weights[5])
        else:
            out = self.layer1(x)
            feat = func.relu(out)
            out = self.last_layer(feat)

        if self.task_incremental:
            scores = [] # get logits of corresponding tasks
            for b in range(x.size(0)):
                scores.append(feat[b, task_ids[b] * self.num_of_classes: (task_ids[b] + 1) * self.num_of_classes])
            out = torch.stack(scores)

        if reduce:
            loss = self.criterion(out, y)
        else:
            loss = self.criterion(out, y, reduction='none')
        return {'score': out, 'loss': loss, 'feat': feat}

    def forward_from_feat(self, feat, y, reduce=True, **kwargs):
        out = self.last_layer[self.ds_idx](feat)
        if reduce:
            loss = self.criterion(out, y)
        else:
            loss = self.criterion(out, y, reduction='none')
        return {'score': out, 'loss': loss, 'feat': feat}



def mnist_simple_net_400width_classlearning_1024input_10cls_1ds(**kwargs):
    return FC2Layers(input_size=28 * 28, layer1_width=400, layer2_width=400, **kwargs)

def mnist_simple_net_400width_domainlearning_1024input_10cls_1ds(**kwargs):
    return FC2Layers(input_size=28 * 28, layer1_width=400, layer2_width=400, **kwargs)