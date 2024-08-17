import json
import os
import shutil
from sklearn.modl_selection import train_test_split
from tqdm import tqdm

########## Split the dataset into training and testing sets: random_state=000000; train_size=0.8 ##########

def load_data(root_dir, json_file):
    with open(os.path.join(root_dir, json_file), 'r') as file:
        data = json.load(file)
    return data

def copy_files(file_paths, new_dir, mod, root_dir):
    new_paths = []
    for path in file_paths:
        basename = os.path.basename(path)
        path = os.path.join(root_dir, path)
        new_path = os.path.join(new_dir, basename)
        shutil.copy(path, new_path)
        new_path = f"{mod}/{basename}"
        new_paths.append(new_path)
    return new_paths

def split_data(json_file, mod, root_dir, output_dir, train_size=0.8,):
    data = load_data(root_dir, json_file)
    
    items = list(data.items())
    train_items, test_items = train_test_split(items, train_size=train_size, random_state=000000)
    
    output_train_images_dir = os.path.join(output_dir, mod, 'train/images')
    output_train_masks_dir = os.path.join(output_dir, mod, 'train/masks')
    output_test_images_dir = os.path.join(output_dir, mod, 'test/images')
    output_test_masks_dir = os.path.join(output_dir, mod, 'test/masks')
    os.makedirs(output_train_images_dir, exist_ok=True)
    os.makedirs(output_train_masks_dir, exist_ok=True)
    os.makedirs(output_test_images_dir, exist_ok=True)
    os.makedirs(output_test_masks_dir, exist_ok=True)

    mod = [f"{mod}/train/images", f"{mod}/train/masks", f"{mod}/test/images", f"{mod}/test/masks"]
    
    train_data = {}
    for image_path, mask_paths in tqdm(train_items):
        new_image_path = copy_files([image_path], output_train_images_dir, mod[0], root_dir)[0]
        new_mask_paths = copy_files(mask_paths, output_train_masks_dir, mod[1], root_dir)
        train_data[new_image_path] = new_mask_paths
    
    with open(os.path.join(output_dir, mod, 'train/image2label_train.json'), 'w') as file:
        json.dump(train_data, file, indent=4)
    
    test_data = {}
    for image_path, mask_paths in tqdm(test_items):
        new_image_path = copy_files([image_path], output_test_images_dir, mod[2])[0]
        new_mask_paths = copy_files(mask_paths, output_test_masks_dir, mod[3])
        test_data[new_image_path] = new_mask_paths

    label2image_test = {}
    for image_path, mask_paths in test_data.items():
        for mask_path in mask_paths:
            label2image_test[mask_path] = image_path

    with open(os.path.join(output_dir, mod, 'test/label2image_test.json'), 'w') as file:
        json.dump(label2image_test, file, indent=4)

if __name__ == '__main__':
    m = [f"ct_{x}" for x in range(1, 12)]
    m += [f"dermoscopy_{x}" for x in range(1, 4)]
    m += [f"mr_{x}" for x in range(1, 3)]
    m += ["endoscopy_1"]
    m += ["pet_1"]
    m += ["x_1"]
    m += ["fundus_1"]
    m += ["ultrasound_1"]
    
    jsons = [f"{x}/{x}_mapping.json" for x in m]
    
    root_dir = '/path/to/processed'
    output_dir = 'path/to/train_test'
    
    for i, file in enumerate(jsons):
        split_data(file, m[i], root_dir, output_dir)
