from model.Net import Net, Loss
from utils.solver import Solver, get_logger
from provider.create_dataloaders import create_dataloaders
import os
import sys
import argparse
import logging
import random

import torch
import gorilla

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'provider'))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'model', 'pointnet2'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))


def get_parser():
    parser = argparse.ArgumentParser(
        description="Pose Estimation")

    # pretrain
    parser.add_argument("--gpus",
                        type=str,
                        default="0",
                        help="gpu num")
    parser.add_argument("--config",
                        type=str,
                        help="path to config file")
    parser.add_argument("--pretrained_model",
                        type=str,
                        help="path to pretrained model")
    parser.add_argument("--finetune",
                        type=bool,
                        default=True,
                        help="finetune")
    args_cfg = parser.parse_args()

    return args_cfg


def init():
    args = get_parser()
    exp_name = args.config.split("/")[-1].split(".")[0]
    log_dir = os.path.join("log", exp_name)

    if not os.path.isdir("log"):
        os.makedirs("log")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    cfg = gorilla.Config.fromfile(args.config)
    cfg.finetune = args.finetune
    cfg.pretrained_model = args.pretrained_model
    cfg.exp_name = exp_name
    cfg.log_dir = log_dir
    cfg.ckpt_dir = os.path.join(log_dir, 'ckpt')
    if not os.path.isdir(cfg.ckpt_dir):
        os.makedirs(cfg.ckpt_dir)

    cfg.gpus = args.gpus
    logger = get_logger(
        level_print=logging.INFO, level_save=logging.WARNING, path_file=log_dir+"/training_logger.log")
    gorilla.utils.set_cuda_visible_devices(gpu_ids=cfg.gpus)

    return logger, cfg


if __name__ == "__main__":
    logger, cfg = init()

    logger.warning(
        "************************ Start Logging ************************")
    logger.info(cfg)
    logger.info("using gpu: {}".format(cfg.gpus))

    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed_all(cfg.rd_seed)

    # model
    logger.info("=> creating model ...")
    model = Net(cfg.pose_net)

    start_epoch = 1
    start_iter = 0

    model = model.cuda()

    # Load pretrained model for finetuning
    if cfg.finetune and cfg.pretrained_model:
        logger.info("=> loading pretrained model from '{}'".format(
            cfg.pretrained_model))
        checkpoint = torch.load(cfg.pretrained_model)
        if 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        else:
            pretrained_dict = checkpoint

        model_dict = model.state_dict()
        # Filter out unnecessary keys
        pretrained_dict = {k: v for k,
                           v in pretrained_dict.items() if k in model_dict}
        # Overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # Load the new state dict
        model.load_state_dict(model_dict)
        logger.info("=> loaded pretrained model '{}' (epoch {})".format(
            cfg.pretrained_model, checkpoint.get('epoch', 'unknown')))
    #############################################################################
    count_parameters = sum(gorilla.parameter_count(model).values())
    logger.warning("#Total parameters : {}".format(count_parameters))
    loss = Loss(cfg.loss).cuda()

    # dataloader
    dataloaders = create_dataloaders(cfg.train_dataset)

    for k in dataloaders.keys():
        dataloaders[k].dataset.reset()

    # solver
    Trainer = Solver(model=model,
                     loss=loss,
                     dataloaders=dataloaders,
                     logger=logger,
                     cfg=cfg,
                     start_epoch=start_epoch,
                     start_iter=start_iter)
    Trainer.solve()

    logger.info('\nFinish!\n')
