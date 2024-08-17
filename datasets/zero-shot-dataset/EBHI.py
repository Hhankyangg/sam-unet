import json
import os
import shutil
import cv2
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm


threshold = 100 / 256 / 256

def copy_files(root_dir, temp_dir):
    temp_image_dir = os.path.join(temp_dir, 'images')
    temp_mask_dir = os.path.join(temp_dir, 'masks')
    os.makedirs(temp_image_dir, exist_ok=True)
    os.makedirs(temp_mask_dir, exist_ok=True)

    for subdir, dirs, files in tqdm(os.walk(root_dir)):
        for file in files:
            if 'image' in subdir:
                src = os.path.join(subdir, file)
                dst = os.path.join(temp_image_dir, file)
                shutil.copy(src, dst)
            elif 'label' in subdir:
                src = os.path.join(subdir, file)
                dst = os.path.join(temp_mask_dir, file) 
                shutil.copy(src, dst)

    print("Files have been copied.")
    

def process_masks(temp_dir, target_dir, json_file):
    temp_image_dir = os.path.join(temp_dir, 'images')
    temp_mask_dir = os.path.join(temp_dir, 'masks')
    target_image_dir = os.path.join(target_dir, 'images')
    target_mask_dir = os.path.join(target_dir, 'masks')
    os.makedirs(target_image_dir, exist_ok=True)
    os.makedirs(target_mask_dir, exist_ok=True)
    label2image = {}
    
    for mask_file in tqdm(os.listdir(temp_mask_dir)):
        if mask_file.endswith('.png'):
            mask_path = os.path.join(temp_mask_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            mask[mask < 150] = 0
            mask[mask >= 150] = 1
            
            image_file = mask_file.replace('mask', 'img')
            image_path = os.path.join(temp_image_dir, image_file)
            shutil.copy(image_path, os.path.join(target_image_dir, image_file))
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            for i in range(1, num_labels):
                individual_mask = (labels == i).astype(int)
                height, width = individual_mask.shape
                ratio = np.sum(individual_mask) / height / width
                if ratio < threshold:
                    continue
                else:
                    output_mask_file = f'{os.path.splitext(mask_file)[0]}_{i}.jpg'
                    output_mask_path = os.path.join(target_mask_dir, output_mask_file)
                    # cv2.imwrite(output_mask_path, individual_mask)
                    plt.imsave(output_mask_path, individual_mask, cmap='gray')
                    label2image[os.path.join('masks', output_mask_file)] = os.path.join('images', image_file)

    with open(json_file, 'w') as f:
        json.dump(label2image, f, indent=4)

def main(root_dir, temp_dir, target_dir, json_file):
    copy_files(root_dir, temp_dir)
    process_masks(temp_dir, target_dir, json_file)

if __name__ == '__main__':
    root_dir = 'path/to/EBHI-SEG'
    temp_dir = 'path/to/EBHI-SEG-Temp'
    target_dir = 'path/to/output'
    json_file = 'path/to/output/label2image_test.json'
    main(root_dir, temp_dir, target_dir, json_file)
