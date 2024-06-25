from ultralytics import utils
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

from ultralytics.models.yolo.model import YOLO

settingmgr = utils.SettingsManager()
DATASETS_DIR = Path(settingmgr.defaults['datasets_dir'])

def test_yaml_training_with_3_channels():
    build_random_seg_dataset(DATASETS_DIR, 'multichannel')
    model = YOLO('yolov8n-seg.yaml')
    model.train(data=DATASETS_DIR / 'multichannel/data.yaml', epochs=1, save=False)
    
def build_random_seg_dataset(root, name):
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
            img = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
            
            label = np.random.random((4)) * 0.5
            label.sort() ## get a xyxy polygon

            Image.fromarray(img).save(dataset / f'images/{phase}/{i}.jpg')
            
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