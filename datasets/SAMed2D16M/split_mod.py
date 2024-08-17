import os
import shutil
import json
from tqdm import tqdm

########## Sort the dataset into separate folders by modality and limit the folder sizes. ##########

def main(root_directory, output_directory, size_limit, json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    modalities_mapping = {}

    size_dict_for_each_modality = {}

    for image_path, mask_paths in tqdm(data.items()):
        
        init_num = 1 
        modality = image_path.split('/')[-1].split('_')[0] + "_" + str(init_num) 
        while size_dict_for_each_modality.get(modality, 0) >= size_limit:
            init_num += 1
            modality = modality.split('_')[0] + "_" + str(init_num)
        
        modality_dir = os.path.join(output_directory, modality)
        
        modality_image_dir = os.path.join(modality_dir, "images")
        modality_mask_dir = os.path.join(modality_dir, "masks")
        os.makedirs(modality_image_dir, exist_ok=True)
        os.makedirs(modality_mask_dir, exist_ok=True)
        
        full_image_path = os.path.join(root_directory, image_path)
        
        if size_dict_for_each_modality.get(modality) is not None:
            size_dict_for_each_modality[modality] += os.path.getsize(full_image_path)
        else:
            size_dict_for_each_modality[modality] = os.path.getsize(full_image_path)
        
        image_name = os.path.basename(image_path)
        shutil.copy(full_image_path, os.path.join(modality_image_dir, image_name))

        image_item = f'{modality}/images/{image_name}'
        modalities_mapping.setdefault(modality, {}).setdefault(image_item, [])

        for mask_path in mask_paths:
            full_mask_path = os.path.join(root_directory, mask_path)
            size_dict_for_each_modality[modality] += os.path.getsize(full_mask_path)
            mask_name = os.path.basename(mask_path)
            shutil.copy(full_mask_path, os.path.join(modality_mask_dir, mask_name))
            mask_item = f'{modality}/masks/{mask_name}'
            modalities_mapping[modality][image_item].append(mask_item)

    for modality, mapping in tqdm(modalities_mapping.items()):
        mapping_path = os.path.join(output_directory, modality, f"{modality}_mapping.json")
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=4, ensure_ascii=False)

    print("All files have been copied and mappings created.")
    

if __name__ == '__main__':
    size_limit=20*1024**3
    json_file = 'path/to/SAMed2Dv1/SAMed2D_v1.json'
    root_directory = "path/to/SAMed2Dv1"
    output_directory = "path/to/processed"
    main(root_directory, output_directory, size_limit, json_file)
