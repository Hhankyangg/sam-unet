import argparse
import torch
import os

from torch.utils.data import DataLoader

from sam_unet.dataset import TestingDataset, custom_collate_fn
from sam_unet.models.build_sam_unet import sam_unet_registry
from sam_unet.utils.utils import get_logger, setup_seeds
from sam_unet.utils.metrics import SegMetrics
from sam_unet.config import config_dict

import time
from tqdm import tqdm
import numpy as np
import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="test_zero_shot", help="run model name")
    parser.add_argument("--batch_size", type=int, default=1, help="train batch size")
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--checkpoint", type=str, default='path/to/tested_model_checkpoint.pth', help="load checkpoint") 
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers in dataloader")
    parser.add_argument("--model_type", type=str, default='Res50', help="'Res50' or 'Res34'")
    args = parser.parse_args()
    return args


def to_device(batch_input, device):
    device_input = []
    for one_dict in batch_input:
        one_dict_device = {}
        for key, value in one_dict.items():
            if key == 'image' or key == 'labels' or key == 'boxes':
                one_dict_device[key] = value.float().to(device)
            else:
                one_dict_device[key] = value
        device_input.append(one_dict_device)
    return device_input


def test_one_epoch(args, model, test_loader, logger):
    
    model.eval()
    test_loader = tqdm(test_loader)
    test_iter_metrics = [0] * len(args.metrics)     # [0, 0]
    illegal_times = 0
    
    for batch, dict_list in enumerate(test_loader):
        batch_size = len(dict_list)                 # len(dict_list) may not always be equal to config["batch_size"]
        dict_list = to_device(dict_list, args.device)
        
        out = model.infer(list_input=dict_list)
        
        for i in range(batch_size): 
            legal_num = 0
            seg_met = []
            if out[i]["masks"].shape == dict_list[i]["labels"].shape:
                legal_num += 1
                seg_met.append(SegMetrics(out[i]["masks"], dict_list[i]["labels"], args.metrics))
        if legal_num == 0:
            illegal_times += 1
            continue
        test_batch_metrics = sum(seg_met) / legal_num

        if int(batch+1) % 50 == 0:
            print(f'Batch: {batch+1}, box prompt: {test_batch_metrics}')
            logger.info(f'Batch: {batch+1}, box prompt: {test_batch_metrics}, test_iter_metrics so far: {[metric / (batch - illegal_times) for metric in test_iter_metrics]}')
            
        test_iter_metrics = [test_iter_metrics[i] + test_batch_metrics[i] for i in range(len(args.metrics))]
    
    return test_iter_metrics, illegal_times


def main(args):
    
    setup_seeds()
    if args.model_type == 'Res50':
        model_type = 'res50_sam_unet'
    else:
        model_type = 'res34_sam_unet'

    model = sam_unet_registry[model_type](need_ori_checkpoint=False, sam_unet_checkpoint=args.checkpoint).to(args.device)
    
    loggers = get_logger(os.path.join(config_dict["work_dir"], "logs", args.run_name + f"_{datetime.datetime.now().strftime('%Y%m%d-%H%M.log')}"))
    loggers.info(f"****{args}****")

    test_dataset = TestingDataset(data_dir=config_dict["data_directory"])
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False, collate_fn=custom_collate_fn)

    l_test = len(test_loader)

    test_iter_metrics, illegal_times = test_one_epoch(args, model, test_loader, loggers)
    test_iter_metrics = [metric / (l_test - illegal_times) for metric in test_iter_metrics]
    test_metrics = {args.metrics[i]: '{:.4f}'.format(test_iter_metrics[i]) for i in range(len(test_iter_metrics))}
    loggers.info(f"Len: {l_test}, Test metrics: {test_metrics}")

    
if __name__ == '__main__':
    args = parse_args()
    main(args)
    