from proposed.data.kitchen.src.kitchen_data_loader import D4RLGCIDDataset
from proposed.prior_train.trainer import get_loader
from easydict import EasyDict as edict
from copy import deepcopy

from proposed.utils import goal_checker
import numpy as np

import torch

args = edict(
    reg_beta = 0.0005,
    epochs = 70,
    mode = "default",
    ga = False,
    min = 30,
    max = 50,
    last = True,
    warmup = 20,
)


args['goal_range'] = (int(args.min), int(args.max))


train_conf, train_loader = get_loader("aa", "train", **args)


dataset = train_loader.dataset

data = dataset[0]


for batch in train_loader:
    break

batch.keys()

torch.where(batch['state_labels'] == 1)[0]
torch.where(batch['state_labels'] == 1)[1]

torch.where()

batch['states'][:, 0].shape

a = torch.randint(0, 2, (3,3))

torch.where(a == 1)


# branching factor인지 아닌지만 판단하면 됨 


batch['states'][:, 0][batch['state_labels'][:, 0] == 1]