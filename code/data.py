import os
import argparse

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SIZE = [256, 512]
DEGREE = 5
BRIGHT_PROB = 0.2
SATURA_PROB = 0.2
CONTRAST_PROB = 0.2
HUE_PROB = 0.2
PADDING = 10


class ImageDataset(Dataset):
    def __init__(self, image_path, label_path):

        self.image_path = image_path
        self.label_path = label_path
        self.label = pd.read_csv(self.label_path, header=None, names=["id", "img_file"])

        # id remapping
        self.id2idx = {}
        for idx, id in enumerate(self.label.id.unique()):
            self.id2idx[id] = idx
        self.idx2id = {v: k for k, v in self.id2idx.items()}
        self.label.id = self.label.id.apply(lambda x: self.id2idx[x])

        # idx 2 all image
        self.idx2image = {}
        for i in range(self.label.shape[0]):
            idx = self.label.id.iloc[i]
            img_name = self.label.img_file.iloc[i]
            if idx not in self.idx2image:
                self.idx2image[idx] = []
            self.idx2image[idx].append(img_name)

        self.transform = T.Compose(
            [
                T.Resize(SIZE),
                T.RandomRotation(DEGREE),
                T.ColorJitter(
                    brightness=BRIGHT_PROB,
                    saturation=SATURA_PROB,
                    contrast=CONTRAST_PROB,
                    hue=HUE_PROB,
                ),
		T.Pad(PADDING),
                T.RandomCrop(SIZE),
                T.ToTensor(),
                T.Normalize(MEAN, STD),
            ]
        )

        self.transform2 = T.Compose(
            [
                T.Resize(SIZE),
                T.RandomRotation(DEGREE),
                T.ColorJitter(
                    brightness=BRIGHT_PROB,
                    saturation=SATURA_PROB,
                    contrast=CONTRAST_PROB,
                    hue=HUE_PROB,
                ),
		T.Pad(PADDING),
                T.RandomCrop(SIZE),
                T.ToTensor(),
                RandomErasing(probability=1.0, mean=MEAN),
                T.Normalize(MEAN, STD),
            ]
        )



    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, index):
        img_fn = self.label.iloc[index].img_file
        path = os.path.join(self.image_path, img_fn)
        image = Image.open(path)
        plain_image = self.transform(image)
        erase_image = self.transform2(image)

        label = self.label.iloc[index].id
        self_label = torch.tensor(label, dtype=torch.long)
        
        
        '''
        # positive example
        pos_fn = random.sample(self.idx2image[label], 1)[0]
        while pos_fn == img_fn:
            pos_fn = random.sample(self.idx2image[label], 1)[0]

        pos_image = Image.open(os.path.join(self.image_path, pos_fn))
        pos_image = self.transform(pos_image)
        pos_label = torch.tensor(label, dtype=torch.long)
        
        # negative example
        neg_idx = random.sample(list(self.idx2id.keys()), 1)[0]
        while neg_idx == label:
            neg_idx = random.sample(list(self.idx2id), 1)[0]

        neg_fn = random.sample(self.idx2image[neg_idx], 1)[0]
        neg_image = Image.open(os.path.join(self.image_path, neg_fn))
        neg_image = self.transform(neg_image)
        neg_label = torch.tensor(neg_idx, dtype=torch.long)
        
        return image, self_label, pos_image, neg_image
        '''
        
        return plain_image, erase_image, self_label


class QueryDataset(Dataset):
    def __init__(
        self, image_path, query_path, gallery_path):
        self.image_path = image_path
        self.query_path = query_path
        self.gallery_path = gallery_path

        self.query = pd.read_csv(query_path, header=None, names=["ans_id", "img_file"])
        self.gallery = pd.read_csv(gallery_path, header=None, names=["id", "img_file"])

        self.transform = T.Compose(
            [
		T.Resize(SIZE),
                T.ToTensor(),
                T.Normalize(MEAN, STD),
            ]
        )
	
        self.transform2 = T.Compose(
            [
                T.Resize(SIZE),
                T.ToTensor(),
                RandomErasing(probability=1.0, mean=MEAN),
                T.Normalize(MEAN, STD),
            ]
        )	

    def __len__(self):
        return self.query.shape[0]

    def __getitem__(self, index):
        d = self.query.iloc[index]
        img_fn = d.img_file
        path = os.path.join(self.image_path, img_fn)
        image = Image.open(path)
        erase_image = self.transform2(image)
        norm_image = self.transform(image)
        
        label = d.ans_id
        self_label = torch.tensor(label, dtype=torch.long)
        
        return norm_image, erase_image, self_label


    def getGalleryTensor(self):

        allGalleryImage = []
        allEraseImage = []
        for i in range(self.gallery.shape[0]):
            img_fn = self.gallery.iloc[i].img_file
            path = os.path.join(self.image_path, img_fn)
            image = Image.open(path)
            norm_image = self.transform(image).unsqueeze(0)
            erase_image = self.transform2(image).unsqueeze(0)
            allGalleryImage.append(norm_image)
            allEraseImage.append(erase_image)

        return allGalleryImage, allEraseImage


import math
import random


class RandomErasing(object):

    def __init__(self, probability=0.5, sl=0.005, sh=0.1, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img
        
        '''
        new_img = Image.new(img.mode, img.size)
        new_img_pixel = []
        '''

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                
                '''
                new_img_pixel = img.load()
                new_img_pixel[x1:x1 + h, y1:y1 + w] = self.mean
                new_img.putdata(new_img_pixel)
                '''
                
                #if img.size[0] == 3:
                img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                #else:
                #    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


'''
if __name__ == "__main__":

    image_dir = "../dataset/aug_img/"
    label_path = "../dataset/train_aug.csv"

    batch_size = 1

    dataset = ImageDataset(image_dir, label_path)
    dataloader = DataLoader(
        dataset, shuffle=True, batch_size=batch_size, num_workers=7
    )
    
    data = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(data[0].to("cpu")[:64], padding=2, normalize=True).cpu(),
            (1, 2, 0),
        ),
    )
    #plt.show()
    #plt.savefig("./test.jpg")
    '''



