import os
import json
from tqdm import tqdm

########## Remove data with incorrect image-mask correspondences from the dataset. ##########

def data_clear(root_path, json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    to_delete_images = []
    num_mask_deleted = 0

    for image_path, masks in tqdm(data.items()):
        _image_path = os.path.join(root_path, image_path)

        if not os.path.exists(_image_path):
            to_delete_images.append(image_path)
        else:
            to_delete_masks = []
            for mask in masks:
                _mask = os.path.join(root_path, mask)
                if not os.path.exists(_mask):
                    num_mask_deleted += 1
                    to_delete_masks.append(mask)
            
            for mask in to_delete_masks:
                data[image_path].remove(mask)
            
            if not data[image_path]:
                to_delete_images.append(image_path)

    print(f"Deleted {len(to_delete_images)} images and {num_mask_deleted} masks.")

    for image in to_delete_images:
        del data[image]
        

    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4)
        
def sum_up(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        
    num_images = len(data)
    num_masks = sum([len(masks) for masks in data.values()])
    
    print(f"Number of images: {num_images}")
    print(f"Number of masks: {num_masks}")
    
    return num_images, num_masks

def main(root_path, json_file_path):
    data_clear(root_path, json_file_path)
    sum_up(json_file_path)
    
if __name__ == '__main__':
    root_path = 'path/to/SAMed2Dv1'
    json_file_path = 'path/to/SAMed2Dv1/SAMed2D_v1.json'
    main(root_path, json_file_path)
