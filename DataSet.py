#imports
import torch
import typing
from torch import nn, optim
from torch._C import INSERT_FOLD_PREPACK_OPS
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, dataset
from typing import List, Dict
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb



class Data_SET:

    # a class to represent our dataset
    # we are going to use Map-style datasets , we should define __len__() and __getitem__() method 
    

    def __init__(self,paths :List , augmentation :bool ,SIZE : int) -> None:

        self.paths=paths
        self.augmentation=augmentation
        self.size=SIZE

        if augmentation:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p =0.5), #data augmentation
                transforms.Resize((SIZE,SIZE),interpolation= Image.BICUBIC) #resize the image
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.SIZE,self.SIZE),interpolation= Image.BICUBIC)
            ])
    
    def __getitem__(self,idx :int) -> Dict:
        if idx >= len(self.paths):
            return Image.NONE
        img = Image.open(self.paths[idx]).convert("RGB") #read image
        img = self.transform(img = img) #transform tha image (resize and flip)
        img = np.array(img) 
        lab_img = rgb2lab(img).astype("float") #convert from rgb to lab
        lab_img = transforms.ToTensor()(lab_img)  # transform from array to tensor
        L = lab_img[[0],...] /50. -1. #normalize values between -1 and 1
        ab = lab_img[[1,2],...] /110. #normalize values between -1 and 1

        return {"L" : L ,"ab" : ab}

    def __len__(self) -> int:
        return len(self.paths)


def create_dataloader(batch_size = 16 ,n_workers = 4 , pin_memory=True,**kwargs):

    #create dataset object
    dataset = Data_SET(**kwargs)
    #create an iterable over dataset

    data_loader = DataLoader(dataset= dataset,
                            batch_size= batch_size,
                            num_workers= n_workers,
                            pin_memory=pin_memory)
    return data_loader
