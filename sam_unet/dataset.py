import os
from torch.utils.data import Dataset
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from sam_unet.utils.utils import get_boxes_from_mask, ResizeLongestSide
import json
import random

from sam_unet.config import config_dict


root = config_dict['root_dir']

class TestingDataset(Dataset):
    
    def __init__(self, data_dir, image_size=config_dict['img_size'], mode='test', requires_name=False):
        """
        Initializes a TestingDataset object.
        Arguments:
            data_dir (str / list): The path to the data.
            image_size (int, optional): The size of the image. Defaults to 256.
            mode (str, optional): The mode of the dataset. Defaults to 'test'.
            requires_name (bool, optional): Indicates whether the dataset requires image names. Defaults to True.
            point_num (int, optional): The number of points to retrieve. Defaults to 1.
        """
        
        self.image_size = image_size
        self.requires_name = requires_name
        if isinstance(data_dir, list):
            self.image_paths = []
            self.label_paths = []
            for dir in data_dir:
                dataset = json.load(open(os.path.join(dir, mode, f'label2image_{mode}.json'), "r"))
                self.image_paths.extend(list(os.path.join(root, x) for x in dataset.values()))
                self.label_paths.extend(list(os.path.join(root, x) for x in dataset.keys()))
        else:
            dataset = json.load(open(os.path.join(data_dir, mode, f'label2image_{mode}.json'), "r"))
            self.image_paths = list(os.path.join(root, x) for x in dataset.values())
            self.label_paths = list(os.path.join(root, x) for x in dataset.keys())
    
    def __getitem__(self, index):
        """
        Retrieves and preprocesses an item from the dataset.
        Arguments:
            index (int): The index of the item to retrieve.
        Returns:
            dict: A dictionary containing the preprocessed image and associated information.
        """
        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # [ori_h, ori_w, 3]
        except:
            print(self.image_paths[index])

        mask_path = self.label_paths[index]
        ori_np_mask = cv2.imread(mask_path, 0)              # [ori_h, ori_w]
        
        if ori_np_mask.max() == 255:
            ori_np_mask = ori_np_mask / 255
        ori_np_mask = ori_np_mask.astype(np.int64)          # [ori_h, ori_w]

        assert np.array_equal(ori_np_mask, ori_np_mask.astype(bool)), f"Mask should only contain binary values 0 and 1. {self.label_paths[index]}"

        h, w, _ = image.shape
        origin_size = (h, w)

        transforms = ResizeLongestSide(self.image_size)
        
        img_resized = transforms.apply_image(image=image)   # [h', w', 3] ndarray
        img_t = torch.as_tensor(img_resized)
        img_t = img_t.permute(2, 0, 1).contiguous()         # [3, h', w']

        boxes = get_boxes_from_mask(ori_np_mask, max_pixel = 0)
        boxes = transforms.apply_boxes_torch(boxes=boxes, original_size=origin_size)

        mask = torch.tensor(ori_np_mask).unsqueeze(0)       # [1, ori_h, ori_w]

        image_input["image"] = img_t                        # [3, h', w']
        image_input["original_size"] = (h, w)
        image_input["labels"] = mask.unsqueeze(0)           # [1, 1, ori_h, ori_w]
        image_input["boxes"] = boxes                            # [1, 4]
        image_input["label_path"] = '/'.join(mask_path.split('/')[:-1])
        image_input["point_coords"] = None
        image_input["point_labels"] = None
        
        if self.requires_name:
            image_name = self.label_paths[index].split('/')[-1]
            image_input["name"] = image_name
            return image_input
        else:
            return image_input

    def __len__(self):
        return len(self.label_paths)


class TrainingDataset(Dataset):

    def __init__(self, data_dir, image_size=config_dict['img_size'], mode='train', requires_name=False, mask_num=5):
        """
        Initializes a training dataset.
        Arguments:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
        """
        
        self.image_size = image_size 
        self.requires_name = requires_name 
        self.mask_num = mask_num 
        if isinstance(data_dir, list):
            self.image_paths = []
            self.label_paths = []
            for dir in data_dir:
                dataset = json.load(open(os.path.join(dir, mode, f'image2label_{mode}.json'), "r"))
                self.image_paths.extend([os.path.join(root, x) for x in dataset.keys()])
                self.label_paths.extend([[os.path.join(root, x) for x in sublist] for sublist in dataset.values()])
        else:
            dataset = json.load(open(os.path.join(data_dir, mode, f'image2label_{mode}.json'), "r"))
            self.image_paths = [os.path.join(root, x) for x in dataset.keys()]
            self.label_paths = [[os.path.join(root, x) for x in sublist] for sublist in dataset.values()]

        
    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Arguments:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """
        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # [ori_h, ori_w, 3]
        except:
            print(self.image_paths[index])

        h, w, _ = image.shape
        origin_size = (h, w)
        transforms = ResizeLongestSide(self.image_size)
        img_resized = transforms.apply_image(image=image)   # [h', w', 3] ndarray
        img_t = torch.as_tensor(img_resized)
        img_t = img_t.permute(2, 0, 1).contiguous()         # [3, h', w']
        
        masks_list = []
        boxes_list = []
        mask_path = random.choices(self.label_paths[index], k=self.mask_num)
        for m in mask_path:
            pre_mask = cv2.imread(m, 0)                     # [ori_h, ori_w]
            if pre_mask.max() == 255:
                pre_mask = pre_mask / 255
            pre_mask = pre_mask.astype(np.uint8)            # [ori_h, ori_w]

            boxes = get_boxes_from_mask(pre_mask)
            boxes = transforms.apply_boxes_torch(boxes=boxes, original_size=origin_size)
            pre_mask = transforms.apply_image(image=pre_mask)   # [h', w'] the longest one in (h', w') is image_size
            h, w = pre_mask.shape[-2:]
            padh = self.image_size - h
            padw = self.image_size - w
            pre_mask = F.pad(torch.from_numpy(pre_mask), (0, padw, 0, padh))
            
            masks_list.append(pre_mask) 
            boxes_list.append(boxes)

        mask = torch.stack(masks_list, dim=0)                   # [mask_num, ori_h, ori_w]
        boxes = torch.stack([i.squeeze(0) for i in boxes_list], dim=0)

        image_input["image"] = img_t                            # [3, h', w'] # the longest one in (h', w') is image_size
        image_input["original_size"] = origin_size              # (ori_h, ori_w)
        image_input["labels"] = mask.unsqueeze(1)               # [mask_num, 1, image_size, image_size]
        image_input["boxes"] = boxes                            # [mask_num, 4]
        image_input["point_coords"] = None
        image_input["point_labels"] = None
        
        
        if self.requires_name:
            image_name = self.image_paths[index].split('/')[-1]
            image_input["name"] = image_name
            return image_input
        else:
            return image_input

    def __len__(self):
        return len(self.image_paths)


def custom_collate_fn(batch):
    return batch


if __name__ == "__main__":
    train_dataset = TrainingDataset("./data_demo")
    data_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)
    for i, dict_list in enumerate(tqdm(data_loader)):
        print(f"The batch {i} in dataloader:")
        for j in range(len(dict_list)):
            print(f"  The item {j} in batch {i}:")
            print(f"    Image shape (after  transform): {dict_list[j]['image'].shape}")
            print(f"    Mask  shape (after  transform & padding): {dict_list[j]['labels'].shape}")
