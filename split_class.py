import os
import shutil
from glob import glob

import pandas as pd
from tqdm import tqdm


def create_folder(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print("Error: Creating directory: " + dir)


train_df = pd.read_csv('data/train_df.csv')
label_list = train_df['label'].unique().tolist()

for i in range(len(label_list)):
    create_folder(f'EDA/{label_list[i]}')

img_lst = glob('data/train/*.png')
for img in tqdm(img_lst):
    label = train_df.loc[train_df['file_name'] == img.split('/')[-1]]['label']
    destination = f'EDA/{label.values[0]}/'
    shutil.copy(img, destination)
