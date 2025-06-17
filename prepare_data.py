import glob
import os
import random
import shutil

print("enter dataset name:")
dataset_name = input()

print("enter class1 name:")
c1 = input()

print("enter class2 name:")
c2 = input()

data_dir = os.path.join("data/", dataset_name)

train_dir_parent = os.path.join(data_dir, "train")
train_dir_c1 = os.path.join(data_dir, "train", c1)
train_dir_c2 = os.path.join(data_dir, "train", c2)
test_dir_c1 = os.path.join(data_dir, "test", c1)
test_dir_c2 = os.path.join(data_dir, "test", c2)
val_dir_c1 = os.path.join(data_dir, "validate", c1)
val_dir_c2 = os.path.join(data_dir, "validate", c2)

for dir in [train_dir_c1, train_dir_c2, test_dir_c1, test_dir_c2, val_dir_c1, val_dir_c2]:
    os.makedirs(dir, exist_ok=True)

# get all filepaths by class and put in list per class 
c1_all = glob.glob(f"{train_dir_parent}/{c1}*")
c2_all = glob.glob(f"{train_dir_parent}/{c2}*")
random.shuffle(c1_all)
random.shuffle(c2_all)

def split_move(imgs, train_ratio, train_dir, val_dir):
    '''split off 20% of the training data to be used for validation during training. move to correct folder'''
    split_idx = int(len(imgs) * train_ratio)
    print(split_idx)
    train_files = imgs[:split_idx]
    val_files = imgs[split_idx:]
    print(val_files)

    for img in train_files:
        shutil.move(img, train_dir)
    for img in val_files:
        shutil.move(img, val_dir)

split_move(c1_all, 0.8, train_dir=train_dir_c1, val_dir=val_dir_c1)
split_move(c2_all, 0.8, train_dir=train_dir_c2, val_dir=val_dir_c2)