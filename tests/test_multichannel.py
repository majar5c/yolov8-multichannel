from ultralytics import utils
import pytest
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

from ultralytics.models.yolo.model import YOLO

settingmgr = utils.SettingsManager()
DATASETS_DIR = Path(settingmgr.defaults['datasets_dir'])

def test_yaml_training_with_3_channels():
    dataset_name = '3channel'
    save_as_image = lambda img, path: Image.fromarray(img).save(f'{path}.jpg')
    
    build_random_seg_dataset(DATASETS_DIR, dataset_name, 3, save_as_image)
    model = YOLO('yolov8n-seg.yaml')
    model.train(data=DATASETS_DIR / f'{dataset_name}/data.yaml', epochs=1, save=False)
    
def test_npy_training_with_3_channels():
    dataset_name = '3channel-npy'
    save_as_npy = lambda img, path: np.save(f'{path}.npy', img)
    
    build_random_seg_dataset(DATASETS_DIR, dataset_name, 3, save_as_npy)
    model = YOLO('yolov8n-seg.yaml')
    model.train(data=DATASETS_DIR / f'{dataset_name}/data.yaml', epochs=1, save=False)




## Helper    
def build_random_seg_dataset(root, name, channel, save_callback):
    """
    Create a random dataset for testing segmentation tasks.
    """
    
    # Create dataset directory
    dataset = Path(root) / name
    
    # Create directories and remove existing files
    shutil.rmtree(dataset, ignore_errors=True)
    (dataset / 'images/train').mkdir(parents=True)
    (dataset / 'images/val').mkdir(parents=True)
    (dataset / 'labels/train').mkdir(parents=True)
    (dataset / 'labels/val').mkdir(parents=True)
    
    # Create 6 random images and labels for yolov8 segmentation
    for phase in ['train', 'val']:
        for i in range(6):
            img = np.random.randint(0, 256, size=(32, 32, channel), dtype=np.uint8)
            
            label = np.random.random((4)) * 0.5
            label.sort() ## get a xyxy polygon
            
            save_callback(img, dataset / f'images/{phase}/{i}')
            
            with open(dataset / f'labels/{phase}/{i}.txt', 'w') as f:
                f.write('0')  ## class
                
                ## bbox
                for j in label:
                    f.write(f' {j}')
                
                ## mask
                for j in label:
                    f.write(f' {j}')
    
    ## Create data.yaml for the dataset
    with open(dataset / 'data.yaml', 'w') as f:
        f.write(f"path: {dataset}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 1\n")
        
test_npy_training_with_3_channels()