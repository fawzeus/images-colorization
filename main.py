import matplotlib.pyplot as plt
from DataSet import Data_SET,create_dataloader
import glob
import numpy as np
from PIL import Image
data_path = r"C:\Users\Asus\.fastai\data\coco_sample/train_sample"
from typing import List
#loading all images paths
paths = glob.glob(data_path + "/*.jpg") 
np.random.seed(50)
paths_subset = np.random.choice(paths, 20_000, replace=False) # choosing 20000 images randomly
rand_idxs = np.random.permutation(20_000) #shuffling dataset
train_idxs = rand_idxs[:16000] # take the first 16000 as training set
val_idxs = rand_idxs[16000:] # take last 4000 as validation set
#train images
train_paths = paths_subset[train_idxs]
#validation images
val_paths = paths_subset[val_idxs]
print(len(train_paths), len(val_paths))

#vusializing some images

def visualize(train_paths: List)-> any:
    _, axes = plt.subplots(5, 5, figsize=(30, 30))
    for ax, img_path in zip(axes.flatten(), train_paths):
        ax.imshow(Image.open(img_path))
        ax.axis("off")
    plt.show()
