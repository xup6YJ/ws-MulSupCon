import os
import torch
import pandas as pd
from PIL import Image, ImageFilter, ImageOps
from torch.utils import data
import torchvision.transforms as trans
import csv

def getData(mode: str, dataname: str):
    name = dataname.lower()

    if dataname == 'CXR14':
        if mode == 'train':
            df = pd.read_csv(f'./data_csv/{name}_train.csv')

        elif mode == 'pretrain':
            df = pd.read_csv(f'./data_csv/{name}_train.csv')

        elif mode == 'test':
            df = pd.read_csv(f'./data_csv/{name}_test.csv')

        elif mode == 'valid':
            df = pd.read_csv(f'./data_csv/{name}_valid.csv')

        path = df['img_path'].tolist()

    elif dataname == 'mimic':
        if mode == 'train':
            df = pd.read_csv(f'./data_csv/{name}_train_PA224.csv')

        elif mode == 'pretrain':
            df = pd.read_csv(f'./data_csv/{name}_train_PA224.csv')

        elif mode == 'test':
            df = pd.read_csv(f'./data_csv/{name}_test_PA224.csv')

        elif mode == 'valid':
            df = pd.read_csv(f'./data_csv/{name}_val_PA224.csv')

        path = df['img_path'].tolist()
        path = [os.path.join('/home/bspubuntu/Documents/Data/mimic-cxr-jpg-2.1.0-resize', p) for p in path]



    label = df.drop(columns='img_path').values.tolist()
    return path, label, df


class NIHChestLoader(data.Dataset):
    def __init__(self, args, root: str, mode: str, classes: int, dataname: str):
        """
        Args:
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.mode = mode
        self.classes = classes
        self.dataname = dataname
        
        self.args = args
        self.colorjitter = args.color_jitter
        self.cv = args.color_jitter_value

        self.img_name, self.label, self.df = getData(mode, dataname)

        # train transform
        train_trains_list = []
        if self.dataname == 'CXR14' :
            train_trains_list.append(trans.Resize((224, 224)))
        
        train_trains_list.extend([
                                    trans.RandomHorizontalFlip(),
                                    trans.RandomRotation(20),
                                    trans.ToTensor()
                                ])
        
        train_trains_list.append(trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        # test transform
        test_trains_list = []
        if self.dataname == 'CXR14' :
            test_trains_list.append(trans.Resize((224, 224)))
        test_trains_list.append(trans.ToTensor())

        test_trains_list.append(trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

            
        if self.mode == 'train' or self.mode == 'pretrain':
            self.transformations = trans.Compose(train_trains_list)
        else:
            self.transformations = trans.Compose(test_trains_list)

        print("> Found %d images..." % (len(self.img_name)))  

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):

        path = os.path.join(self.root, self.img_name[index])
        img = Image.open(path).convert('RGB')
        target = torch.tensor(self.label[index], dtype=torch.float32)

        if self.mode == 'pretrain': 
            if self.colorjitter and torch.rand(1) > 0.5:
                img1 = self.transformations1(img)
            else:
                img1 = self.transformations(img)
            img2 = self.transformations(img)

            return [img1, img2], target
        else:
            img = self.transformations(img)
            return img, target

    def get_labels(self):
        return self.label
    

