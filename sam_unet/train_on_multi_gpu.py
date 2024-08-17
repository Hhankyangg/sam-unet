import argparse
import torch
import os

from torch import optim
from torch.utils.data import DataLoader

from sam_unet.dataset import TrainingDataset, TestingDataset, custom_collate_fn
from sam_unet.models.build_sam_unet import sam_unet_registry
from sam_unet.utils.loss import FocalDiceloss
from sam_unet.utils.utils import get_logger, setup_seeds
from sam_unet.utils.metrics import SegMetrics
from sam_unet.config import config_dict

import time
from tqdm import tqdm
import numpy as np
import datetime

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist
from torch.distributed import init_process_group, destroy_process_group


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="train", help="run model name")
    parser.add_argument("--epochs", type=int, default=12, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=50, help="train batch size")
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--resume", type=str, default=None, help="load resume") 
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers in dataloader")
    
    # multi gpu settings
    default_rank = int(os.getenv('LOCAL_RANK', 0))
    parser.add_argument("--local_rank", type=int, default=default_rank)
    
    args = parser.parse_args()
    return args


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def to_device(batch_input, device):
    device_input = []
    for one_dict in batch_input:
        one_dict_device = {}
        for key, value in one_dict.items():
            if key == 'image' or key == 'labels' or key == 'boxes':
                one_dict_device[key] = value.float().to(device)
            # elif key == 'boxes':
            #     one_dict_device[key] = value.to(device)
            else:
                one_dict_device[key] = value
        device_input.append(one_dict_device)
    return device_input
            

def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, loggers):
    train_loader.sampler.set_epoch(epoch + 1)
    model.train()
    
    if args.local_rank == 0:
        train_loader = tqdm(train_loader)
    
    train_losses = []
    train_iter_metrics = [0] * len(args.metrics)
    
    for batch, dict_list in enumerate(train_loader):
        
        batch_size = len(dict_list)     # len(dict_list) may not always be equal to config["batch_size"]
        dict_list = to_device(dict_list, args.local_rank)
        
        out = model(list_input=dict_list)
        masks_cat = torch.cat([d['masks'] for d in out], dim=0)             # [batch_size * mask_num, 1, img_size(256), img_size]
        labels_cat = torch.cat([d['labels'] for d in dict_list], dim=0)     # [batch_size * mask_num, 1, img_size(256), img_size]
        focal_loss, dice_loss = criterion(masks_cat, labels_cat)
        loss = focal_loss + 20 * dice_loss
        loss.backward()
        
        optimizer.step()
        loss = reduce_mean(loss, dist.get_world_size())
        optimizer.zero_grad()
        
        if args.local_rank == 0:
            train_batch_metrics = sum([SegMetrics(out[i]["masks"], dict_list[i]["labels"], args.metrics) for i in range(batch_size)]) / batch_size
            if int(batch+1) % 50 == 0 or batch == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch+1}, box prompt: {train_batch_metrics}')
                loggers.info(f'Epoch: {epoch+1}, Batch: {batch+1}, Loss: {loss.item()}, Focal_loss: {focal_loss.item()}, Dice_loss: {dice_loss.item()}, box prompt: {train_batch_metrics}')
            train_iter_metrics = [train_iter_metrics[i] + train_batch_metrics[i] for i in range(len(args.metrics))]
            train_losses.append(loss.item())
        
            gpu_info = {}
            gpu_info['gpu_name'] = args.local_rank
            train_loader.set_postfix(train_loss=loss.item(), gpu_info=gpu_info) 
    
    return train_losses, train_iter_metrics


def test_one_epoch(args, model, test_loader, epoch,):
    
    model.eval()
    test_loader = tqdm(test_loader)
    test_iter_metrics = [0] * len(args.metrics)     # [0, 0]
    
    for batch, dict_list in enumerate(test_loader):
        batch_size = len(dict_list)                 # len(dict_list) may not always be equal to config["batch_size"]
        dict_list = to_device(dict_list, args.local_rank)

        out = model(list_input=dict_list)
        
        test_batch_metrics = sum([SegMetrics(out[i]["masks"], dict_list[i]["labels"], args.metrics) for i in range(batch_size)]) / batch_size
        if int(batch+1) % 50 == 0:
            print(f'Epoch: {epoch+1}, Batch: {batch+1}, box prompt: {test_batch_metrics}')
        test_iter_metrics = [test_iter_metrics[i] + test_batch_metrics[i] for i in range(len(args.metrics))]
    
    return test_iter_metrics


def main(args):
    
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)
    torch.distributed.init_process_group(
        backend='nccl',
        # init_method='env://',
        timeout=datetime.timedelta(minutes=10) 
    )

    setup_seeds()
    # model
    model = sam_unet_registry['res34_sam_unet'](need_ori_checkpoint=True, sam_unet_checkpoint=None)
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = FocalDiceloss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma = 0.5)
    
    if args.resume is not None:
        torch.distributed.barrier()  
        with open(args.resume, "rb") as f:
            map_location = {'cuda:0': f'cuda:{args.local_rank}'}  
            checkpoint = torch.load(f, map_location=map_location)  
            model.load_state_dict(checkpoint['model'])  
            optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
            print(f"*******load {args.resume}")
    
    train_dataset = TrainingDataset(data_dir=config_dict["data_directory"])
    test_dataset = TestingDataset(data_dir=config_dict["data_directory"])
    train_loader = DataLoader(dataset=train_dataset, 
                             batch_size=args.batch_size, 
                             num_workers=args.num_workers, 
                             shuffle=False, 
                             collate_fn=custom_collate_fn,
                             sampler=DistributedSampler(train_dataset))
    # test_loader = DataLoader(dataset=test_dataset,
    #                          batch_size=args.batch_size,
    #                          num_workers=args.num_workers,
    #                          shuffle=False, collate_fn=custom_collate_fn)

    loggers = None
    if args.local_rank == 0:
        loggers = get_logger(os.path.join(config_dict["work_dir"], "logs", args.run_name + f"_{datetime.datetime.now().strftime('%Y%m%d-%H%M.log')}"))
        loggers.info(f"****{args}****")

        best_loss = 1e10
        l_train = len(train_loader)

    for epoch in range(0, args.epochs):
        if args.local_rank == 0:
            train_metrics = {}
            # test_metrics = {}
            os.makedirs(os.path.join(f"{config_dict['work_dir']}/model_train", args.run_name), exist_ok=True)
        
        train_losses, train_iter_metrics = train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, loggers)
        
        scheduler.step()
        dist.barrier()
        
        if args.local_rank == 0:
            train_iter_metrics = [metric / l_train for metric in train_iter_metrics]  # average over all batches
            train_metrics = {args.metrics[i]: '{:.4f}'.format(train_iter_metrics[i]) for i in range(len(train_iter_metrics))}
            # result: {'iou': 0.xxx, 'dice': 0.yyy}
        
            average_loss = np.mean(train_losses)
            lr = scheduler.get_last_lr()[0] # if args.lr_scheduler is not None else args.lr
            loggers.info(f"Train: epoch: {epoch + 1}, lr: {lr}, Train loss: {average_loss:.4f}, Train metrics: {train_metrics}")

            # test_iter_metrics = test_one_epoch(args, model, test_loader, epoch)
            # test_iter_metrics = [metric / l_test for metric in test_iter_metrics]
            # test_metrics = {args.metrics[i]: '{:.4f}'.format(test_iter_metrics[i]) for i in range(len(test_iter_metrics))}
            # loggers.info(f"Test : epoch: {epoch + 1}, Test metrics: {test_metrics}")
        
            if average_loss < best_loss:
                best_loss = average_loss
                loggers.info(f"****BEST Train epoch: {epoch + 1}, Train metrics: {train_metrics}, Train loss: {average_loss}****")
                save_path = os.path.join(config_dict["work_dir"], "model_train", args.run_name, f"epoch{epoch+1}_sam_unet.pth")
                state = {'model': model.module.float().state_dict(), 'optimizer': optimizer}
                torch.save(state, save_path)
        dist.barrier()

    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3"
    args = parse_args()
    # torch.backends.cudnn.enabled = False
    main(args)
    destroy_process_group()
    