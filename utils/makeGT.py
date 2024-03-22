import os
import shutil
from tqdm import tqdm

data_path = '/mnt/A/satelite/datasets/test/'
imgNames = ["img_5.png", "img_6.png", "img_7.png", "img_8.png", "img_9.png", "img_10.png", "img_11.png", "img_12.png"]
GT_path = '/home/ices/yl/SatelliteSP/results/GT'

def create_GT(path=GT_path):
    sample_dirs = os.listdir(data_path)
    for d in sample_dirs:
        print("dir:", d)
        sample_dir = os.path.join(data_path, d)
        dst_dir = os.path.join(GT_path, d)
        os.makedirs(dst_dir, exist_ok=True)
        for imn in tqdm(imgNames):
            img_path = os.path.join(sample_dir, imn)
            dst_path = os.path.join(dst_dir, imn)
            shutil.copy2(img_path, dst_path)
    
if __name__ == "__main__":
    create_GT(GT_path)