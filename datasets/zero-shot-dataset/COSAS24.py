import os
import cv2
import json
import numpy as np


def main(root_dir, target_dir):
    image_dir = os.path.join(root_dir, 'image')
    mask_dir = os.path.join(root_dir, 'mask')
    output_mask_dir = os.path.join(target_dir, 'masks')
    output_image_dir = os.path.join(target_dir, 'images')
    json_file = os.path.join(target_dir, 'label2image_test.json')
    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)

    label2image = {}

    for mask_file in os.listdir(mask_dir):
        if mask_file.endswith('.png'):
            mask_path = os.path.join(mask_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            image_file = mask_file
            image_path = os.path.join(image_dir, image_file)
            image = cv2.imread(image_path)
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            for i in range(1, num_labels):
                individual_mask = (labels == i).astype(int)
                
                output_mask_file = f'{os.path.splitext(mask_file)[0]}_{i}.png'
                output_mask_path = os.path.join(output_mask_dir, output_mask_file)
                cv2.imwrite(output_mask_path, individual_mask)
                
                label2image[os.path.join(output_mask_dir, output_mask_file)] = image_path

    with open(json_file, 'w') as f:
        json.dump(label2image, f, indent=4)

    print(f"Processing complete. Results saved in {output_mask_dir} and {json_file}")
    
if __name__ == '__main__':
    root_dir = 'path/to/COSAS24'
    target_dir = 'path/to/output'
    main(root_dir, target_dir)
