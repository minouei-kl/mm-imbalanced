
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
import random
from torch.utils.tensorboard import SummaryWriter
# from otemodel import Model
from NaiveCatModel import Model
from dataset import RvlDataset
from train_val import *
from Config import config
from utils import *


if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))


class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


def get_balanced_sampler(labels,  replacement=True):

    num_samples = len(labels)
    labels = torch.LongTensor(np.array(labels))
    class_count = torch.bincount(labels).to(dtype=torch.float)
    class_weighting = 1. / class_count
    sample_weights = class_weighting[labels]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=num_samples, replacement=replacement)
    return sampler


def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


batch_size = 32

train_dataset = RvlDataset(tar_path=config.tar_path)

dataset_val = RvlDataset(
    tar_path=config.tar_path.replace('train', 'test'))


train_sampler = get_balanced_sampler(train_dataset.targets)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,  # shuffle=True,
                              pin_memory=True, num_workers=10)

val_dataloader = DataLoader(dataset_val, batch_size=batch_size,
                            shuffle=False, pin_memory=True, num_workers=10)


model = Model(config)

final_model = model.to(device)


parameter_dict_opt = {'l_r': 2e-5, 'eps': 1e-8}

# Create the optimizer
optimizer = AdamW(final_model.parameters(),
                  lr=parameter_dict_opt['l_r'],
                  eps=parameter_dict_opt['eps'])

# Total number of training steps
# total_steps = len(train_dataloader) * config.epoch
# Set up the learning rate scheduler
# scheduler = get_linear_schedule_with_warmup(optimizer,
#                                             num_warmup_steps=1000,  # Default value
#                                             num_training_steps=total_steps)

scheduler = get_cosine_schedule_with_warmup(
    optimizer, config.epoch, len(train_dataloader))


# Instantiate the tensorboard summary writer
writer = SummaryWriter('runs/final')

# cls_num_list = [19997, 14712, 10823, 7962, 5857, 4306, 3167,
#                 2331, 1715, 1261, 928, 682, 502, 369, 271, 200]
# per_cls_weights = 1.0 / np.array(cls_num_list)
# per_cls_weights = per_cls_weights / \
#     np.sum(per_cls_weights) * len(cls_num_list)
# per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
loss_fn = nn.CrossEntropyLoss()
# loss_fn = FocalLoss()

train(model=final_model, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler,
      train_dataloader=train_dataloader, val_dataloader=val_dataloader,
      epochs=config.epoch, evaluation=True, device=device, save_best=True,
      file_path='./saved_models/best_model.pt', writer=writer)


# flag Class Accuracy: [0.909, 0.889, 0.977, 0.953, 0.860, 0.853, 0.886, 0.906, 0.949, 0.825, 0.818, 0.817,
#                       0.698, 0.760, 0.950, 0.825]
# 4 | - | 0.590614 | 0.709787 | 86.78 | 793.41
