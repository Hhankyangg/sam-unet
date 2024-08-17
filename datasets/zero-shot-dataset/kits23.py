import os
import json
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

slice_threshold = 100 / 256 / 256

def process_mask(mask_data, depth, height, width, slice_threshold=slice_threshold):
    n_slices = mask_data.shape[0]
    retained_slices = []
    for i in tqdm(range(n_slices)):
        slice_data = mask_data[i, :, :]
        slice_data = slice_data.astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(slice_data, connectivity=8)
        for label in range(1, num_labels):
            component_mask = (labels == label).astype(int)
            ratio = np.sum(component_mask) / height / width
            if ratio > slice_threshold:
                retained_slices.append(i)
    return retained_slices

def normalize_and_save_image(img_data, retained_slices, case_folder, output_dir):
    for i in retained_slices:
        img_slice = img_data[i, :, :]
        img_normalized = ((img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice))) * 255
        img_normalized = img_normalized.astype(np.uint8)
        img_rgb = np.stack([img_normalized]*3, axis=-1)
        img_filename = f"{case_folder}_slice_{i}.png"
        plt.imsave(os.path.join(output_dir, 'images', img_filename), img_rgb)

def save_mask_slices(mask_data, retained_slices, organ_name, case_folder, output_dir):
    for i in retained_slices:
        mask_slice = mask_data[i, :, :].astype(int)
        mask_filename = f"{case_folder}_slice_{i}_{organ_name}.png"
        plt.imsave(os.path.join(output_dir, 'masks', mask_filename), mask_slice, cmap='gray')
        yield mask_filename, f"{case_folder}_slice_{i}.png"

def main(root_dir, target_dir):
    mapping = {}
    for case_folder in tqdm(sorted(os.listdir(root_dir))):
    # for case_folder in ['case_00000']:
        case_path = os.path.join(root_dir, case_folder)
        if not os.path.isdir(case_path):
            continue
        imaging_path = os.path.join(case_path, 'imaging.nii.gz')
        img_nii = nib.load(imaging_path)
        img_data = img_nii.get_fdata()

        instances_dir = os.path.join(case_path, 'instances')
        idx = 1
        for instance_file in sorted(os.listdir(instances_dir)):
            if 'annotation-1' in instance_file:
                organ_name = f"{instance_file.split('_')[0]}-{idx}"
                mask_path = os.path.join(instances_dir, instance_file)
                mask_nii = nib.load(mask_path)
                mask_data = mask_nii.get_fdata()

                depth, height, width = mask_data.shape


                retained_slices = process_mask(mask_data, depth, height, width)
                normalize_and_save_image(img_data, retained_slices, case_folder, target_dir)
                for mask_filename, img_filename in save_mask_slices(mask_data, retained_slices, organ_name, case_folder, target_dir):
                    mapping[os.path.join('masks', mask_filename)] = os.path.join('images', img_filename)
                
                idx += 1

    with open(os.path.join(target_dir, 'label2image_test.json'), 'w') as f:
        json.dump(mapping, f, indent=4)

if __name__ == '__main__':
    root_dir = 'path/to/kits23/dataset'
    target_dir = 'path/to/output'
    os.makedirs(os.path.join(target_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'masks'), exist_ok=True)
    main(root_dir, target_dir)
