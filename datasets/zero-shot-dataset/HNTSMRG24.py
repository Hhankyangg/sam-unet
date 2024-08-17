import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import json

def get_subdirectories(directory):
    items = os.listdir(directory)
    subdirectories = [item for item in items if os.path.isdir(os.path.join(directory, item))]
    return subdirectories

def main(root_dir, target_dir):
    dir_list = get_subdirectories(root_dir)

    label2img = {} 
    standard_ratio = 100 / 256 / 256

    for dir in tqdm(dir_list):
        dir_path = os.path.join(root_dir, dir, 'preRT')
        mask_path = os.path.join(dir_path, f'{dir}_preRT_mask.nii.gz')
        img_path = os.path.join(dir_path, f'{dir}_preRT_T2.nii.gz')


        mask_nii = nib.load(mask_path)
        img_nii = nib.load(img_path)

        mask_data = mask_nii.get_fdata()
        img_data = img_nii.get_fdata()

        height, width, depth = mask_data.shape

        retained_slices = []

        for slice_index in range(depth):
            slice_mask = mask_data[:, :, slice_index]
            slice_mask[slice_mask == 2] = 0
            num_pixels_one = np.sum(slice_mask == 1)
            valid_ratio = num_pixels_one / (height * width)

            if valid_ratio > standard_ratio:
                retained_slices.append(slice_index)
                slice_mask = slice_mask.astype(np.uint8)
                mask_name = f'masks/{mask_path.split("/")[-1].split(".")[0]}_slice_{slice_index}.png'
                img_name = f'images/{img_path.split("/")[-1].split(".")[0]}_slice_{slice_index}.png'
                plt.imsave(os.path.join(target_dir, mask_name), slice_mask, cmap='gray')
                label2img[mask_name] = img_name

        for slice_index in retained_slices:
            img_slice = img_data[:, :, slice_index]

            x_min = img_slice.min()
            x_max = img_slice.max()
            img_normalized = ((img_slice - x_min) / (x_max - x_min)) * 255
            img_normalized = img_normalized.astype(np.uint8)

            img_rgb = np.stack([img_normalized]*3, axis=-1)
            img_name = f'images/{img_path.split("/")[-1].split(".")[0]}_slice_{slice_index}.png'
            plt.imsave(os.path.join(target_dir, img_name), img_rgb)
            
    json_path = os.path.join(target_dir, 'label2image.json')
    with open(json_path, 'w') as f:
        json.dump(label2img, f)

    print("JSON mapping file has been saved.")
    
if __name__ == '__main__':
    root_dir = 'path/to/HNTSMRG24_train'
    target_dir = 'path/to/output'
    main(root_dir, target_dir)
