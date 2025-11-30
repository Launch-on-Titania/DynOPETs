import os
import sys
import argparse
import logging
import random
from pathlib import Path
import torch
import gorilla
from rich import print

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "provider"))
sys.path.append(os.path.join(BASE_DIR, "model"))
sys.path.append(os.path.join(BASE_DIR, "model", "pointnet2"))
sys.path.append(os.path.join(BASE_DIR, "utils"))

from model.Net import Net
from utils.solver import get_logger, test_func_COPE119
from provider.nocs_dataset import TestDataset, TestCOPE119Dataset
from utils.evaluation_utils import evaluate


def get_parser():
    parser = argparse.ArgumentParser(description="Pose Estimation")

    # pretrain
    parser.add_argument("--gpus", type=str, default="0", help="gpu num")
    parser.add_argument("--config", type=str, help="path to config file", default="config/COPE119/COPE119.yaml")
    parser.add_argument("--test_epoch", type=int, default=30, help="test epoch")
    parser.add_argument("--scene_id", type=str, default="00", help="scene id")
    parser.add_argument("--category_id", type=str, choices=["bottle", "bowl", "camera", "can", "laptop", "mug"], default="bottle")
    parser.add_argument(
        "--mask_label",
        action="store_true",
        default=True,
        help="whether having mask labels of real data",
    )
    parser.add_argument(
        "--only_eval",
        action="store_true",
        default=False,
        help="whether directly evaluating the results",
    )
    parser.add_argument("--checkpoint", type=str, help="checkpoint path", required=True)
    parser.add_argument("--data_dir", type=str, required=True, help="Path to COPE119_Data directory")

    parser.add_argument("--save_path", "-s", type=str, default="COPE119testepoch2")
    args_cfg = parser.parse_args()
    return args_cfg

category_id_to_cat_id = {
    "bottle": 1,
    "bowl": 2,
    "camera": 3,
    "can": 4,
    "laptop": 5,
    "mug": 6
}

def init():
    args = get_parser()
    exp_name = args.config.split("/")[-1].split(".")[0]
    log_dir = os.path.join("log", exp_name)

    cfg = gorilla.Config.fromfile(args.config)
    cfg.exp_name = exp_name
    cfg.log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)
    cfg.ckpt_dir = os.path.join(cfg.log_dir, "ckpt")
    cfg.gpus = args.gpus
    cfg.test_epoch = args.test_epoch
    cfg.mask_label = args.mask_label
    cfg.only_eval = args.only_eval
    cfg.cat_id = category_id_to_cat_id[args.category_id]
    cfg.save_path = args.save_path
    cfg.checkpoint = args.checkpoint
    cfg.test_dataset.dataset_dir = args.data_dir + f"/{args.category_id}/{args.category_id}_{args.scene_id}"

    # gorilla.utils.set_cuda_visible_devices(gpu_ids=cfg.gpus)
    logger = get_logger(
        level_print=logging.INFO,
        level_save=logging.WARNING,
        path_file=log_dir + "/test_epoch" + str(cfg.test_epoch) + "_logger.log",
    )
    print(cfg)
    return logger, cfg


if __name__ == "__main__":
    logger, cfg = init()

    logger.warning("************************ Start Logging ************************")
    logger.info(cfg)
    logger.info("using gpu: {}".format(cfg.gpus))



    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)

    save_path = cfg.save_path
    os.makedirs(save_path, exist_ok=True)
    if not cfg.only_eval:
        # model
        logger.info("=> creating model ...")
        model = Net(cfg.pose_net)
        model = model.cuda()

        checkpoint = cfg.checkpoint
        logger.info("=> loading checkpoint from path: {} ...".format(checkpoint))
        gorilla.solver.load_checkpoint(model=model, filename=checkpoint)

        # data loader
        dataset = TestCOPE119Dataset(
            cfg.test_dataset.img_size,
            cfg.test_dataset.sample_num,
            cfg.test_dataset.dataset_dir,
            cfg.setting,
            cfg.cat_id,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False
        )
        test_func_COPE119(model, dataloader, save_path)

    # evaluate(save_path, logger, cat_id=cfg.cat_id)
