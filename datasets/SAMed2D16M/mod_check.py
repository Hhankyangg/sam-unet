import os
import json
from tqdm import tqdm

########## Obtain the data volume for each modality in the dataset. ##########

def main():
    with open(json_file, 'r') as file:
        data = json.load(file)

    modalities_cnt_images = {}
    modalities_cnt_masks = {}


    for image_path, mask_paths in tqdm(data.items()):
        
        modality = image_path.split('/')[-1].split('_')[0]
        
        if modalities_cnt_images.get(modality, 0) == 0:
            modalities_cnt_images[modality] = 1
        else:
            modalities_cnt_images[modality] += 1
        
        for mask in mask_paths:
            if modalities_cnt_masks.get(modality, 0) == 0:
                modalities_cnt_masks[modality] = 1
            else:
                modalities_cnt_masks[modality] += 1

    print(f"Images: {modalities_cnt_images}")
    print(f"Masks : {modalities_cnt_masks}")
    
if __name__ == '__main__':
    json_file = 'path/to/SAMed2Dv1/SAMed2D_v1.json'
    main(json_file)
