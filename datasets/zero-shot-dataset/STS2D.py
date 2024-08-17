import os
import json
import shutil
import cv2
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

threshold = 100 / 256 / 256

def main(root_dir, target_dir, label2image):

    mapping = {}
    images_dir = os.path.join(root_dir, 'image')
    masks_dir = os.path.join(root_dir, 'mask')
    target_mask_dir = os.path.join(target_dir, 'masks')
    target_image_dir = os.path.join(target_dir, 'images')
    os.makedirs(target_image_dir, exist_ok=True)
    os.makedirs(target_mask_dir, exist_ok=True)

    for mask_filename in tqdm(os.listdir(masks_dir)):
        mask_path = os.path.join(masks_dir, mask_filename)
        image_path = os.path.join(images_dir, mask_filename)
        shutil.copy(image_path, os.path.join(target_image_dir, mask_filename))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask < 150] = 0
        mask[mask >= 150] = 1
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num_labels):
            individual_mask = (labels == i).astype(int)
            height, width = individual_mask.shape
            ratio = np.sum(individual_mask) / height / width
            if ratio < threshold:
                continue
            else:
                output_mask_file = f'{mask_filename}_{i}.png'
                output_mask_path = os.path.join(target_mask_dir, output_mask_file)
                # cv2.imwrite(output_mask_path, individual_mask)
                plt.imsave(output_mask_path, individual_mask, cmap='gray')
                mapping[os.path.join('masks', output_mask_file)] = os.path.join('images', mask_filename)

    with open(label2image, 'w') as json_file:
        json.dump(mapping, json_file, indent=4)

    print("JSON mapping file has been saved.")
    
if __name__ == '__main__':
    root_dir = 'path/to/STS2D'
    target_dir = 'path/to/output'
    label2image = 'path/to/output/label2image_test.json'
    main(root_dir, target_dir, label2image)
