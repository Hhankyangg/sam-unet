import cv2
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm

organ_map = {1: "pancreas", 2: "kidney", 3: "liver"}

def process_annotations(annotation_data, height, width, depth, folder, threshold = 100 / 256 / 256):
    binary_masks = []
    retained_slices = []
    annotation_data = annotation_data.astype(np.uint8)
    for slice_index in tqdm(range(depth)):
        slice_data = annotation_data[:, :, slice_index]
        unique_values = np.unique(slice_data)
        for value in unique_values:
            if value == 0:
                continue
            binary_mask = (slice_data == value).astype(np.uint8) 
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            
            for label in range(1, num_labels):
                component_mask = (labels == label).astype(int)
                valid_ratio = np.sum(component_mask) / (height * width)
                if valid_ratio > threshold:
                    retained_slices.append(slice_index)
                    organ_name = organ_map[value] if value in organ_map else f'unknown_{value}'
                    mask_filename = f'{folder}_slice_{slice_index}_{organ_name}_{label}.png'
                    binary_masks.append((component_mask, mask_filename))
                    
    return binary_masks, retained_slices

def save_images(image_data, slice_indices, base_filename, output_dir):
    for slice_index in slice_indices:
        img_slice = image_data[:, :, slice_index]
        x_min, x_max = img_slice.min(), img_slice.max()
        img_normalized = ((img_slice - x_min) / (x_max - x_min)) * 255
        img_normalized = img_normalized.astype(np.uint8)
        img_rgb = np.stack([img_normalized]*3, axis=-1)
        plt.imsave(os.path.join(output_dir, 'images', f'{base_filename}_slice_{slice_index}.png'), img_rgb)

def main(root_dir, target_dir):
    mapping = {}
    for folder in tqdm(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        img_path = os.path.join(folder_path, 'image.nii.gz')
        mask_path = os.path.join(folder_path, 'annotation_1.nii.gz')

        img_nii = nib.load(img_path)
        mask_nii = nib.load(mask_path)

        img_data = img_nii.get_fdata()
        mask_data = mask_nii.get_fdata()

        height, width, depth = mask_data.shape

        binary_masks, retained_slices = process_annotations(mask_data, height, width, depth, folder)
        save_images(img_data, retained_slices, folder, target_dir)

        for mask, filename in binary_masks:
            plt.imsave(os.path.join(target_dir, 'masks', filename), mask, cmap='gray')
            mapping[os.path.join('masks', filename)] = os.path.join('images', f'{folder}_slice_{filename.split("_")[2]}.png')

    with open(os.path.join(target_dir, 'label2image_test.json'), 'w') as f:
        json.dump(mapping, f, indent=4)

if __name__ == '__main__':
    root_dir = 'path/to/CURVAS'
    target_dir = 'path/to/output'
    main(root_dir, target_dir)
