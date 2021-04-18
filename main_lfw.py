import os
import json
import random
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import imgaug
import torch
import torch.multiprocessing as mp
import numpy as np
import argparse


seed = 1234
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import models
from models import segmentation


def main(config_path, pretrain):
    """
    This is the main function to make the training up

    Parameters:
    -----------
    config_path : srt
        path to config file
    """
    # load configs and set random seed
    configs = json.load(open(config_path))
    configs["cwd"] = os.getcwd()

    # load model and data_loader
    model = get_model(configs, pretrain)

    train_set, val_set, test_set = get_dataset(configs)

    # init trainer and make a training
    # from trainers.fer2013_trainer import FER2013Trainer
    #from trainers.tta_trainer import FER2013Trainer
    from trainers.tta_trainer import LFWTrainer

    # from trainers.centerloss_trainer import FER2013Trainer
    trainer = LFWTrainer(model, train_set, val_set, test_set, configs, pretrain)

    if configs["distributed"] == 1:
        ngpus = torch.cuda.device_count()
        mp.spawn(trainer.train, nprocs=ngpus, args=())
    else:
        trainer.train()


def get_model(configs,pretrain):
    """
    This function get raw models from models package

    Parameters:
    ------------
    configs : dict
        configs dictionary
    """
    try:
        return models.__dict__[configs["arch"]]
    except KeyError:
        return segmentation.__dict__[configs["arch"]]


def get_dataset(configs):
    """
    This function get raw dataset
    """
    from utils.datasets.lfw_dataset import lfw

    # todo: add transform
    train_set = lfw("train", configs,augment=True)
    val_set = lfw("val", configs)
    test_set = lfw("test", configs)
    return train_set, val_set, test_set


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", help="path to config file")
    argparser.add_argument("--pretrain", type=bool, help="if True, use pretrained version of model. Default=True",default=True)
    args = argparser.parse_args()
    main(args.config,args.pretrain)
