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

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SIZE = [512, 512]
DEGREE = 10
BRIGHT_PROB = 0.2
SATURA_PROB = 0.2
CONTRAST_PROB = 0.2
HUE_PROB = 0.2
PADDING = 10


class ImageDataset(Dataset):
    def __init__(
        self, image_path, label_path, gallery_path=None, train=True, transform=None
    ):
        self.train = train
        self.image_path = image_path
        self.label_path = label_path
        self.label = pd.read_csv(self.label_path, header=None, names=["id", "img_file"])

        if not self.train:
            self.gallery_path = gallery_path
        else:
            # id remapping
            self.id2idx = {}
            for idx, id in enumerate(self.label.id.unique()):
                self.id2idx[id] = idx
            self.idx2id = {v: k for k, v in self.id2idx.items()}
            self.label.id = self.label.id.apply(lambda x: self.id2idx[x])

        if transform is not None:
            self.transform = transform
        else:
            if self.train:
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
            else:
                self.transform = T.Compose(
                    [T.Resize(SIZE), T.ToTensor(), T.Normalize(MEAN, STD)]
                )

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, index):
        path = os.path.join(self.image_path, self.label.iloc[index].img_file)
        image = Image.open(path)
        image = self.transform(image)
        label = torch.tensor(self.label.iloc[index].id, dtype=torch.long)

        return image, label

    def get_num_classes(self):
        return len(self.id2idx)

    def get_gallery(self):
        data = pd.read_csv(self.gallery_path, header=None, names=["id", "img_file"])
        labels = torch.from_numpy(data.id.values)
        images = []
        for image in data.img_file:
            image = Image.open(os.path.join(self.image_path, image))
            image = self.transform(image)
            images.append(image)
        images = torch.stack(images)

        return images, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiger Re-ID dataset.")
    parser.add_argument("image_dir", type=str, help="Path to image directory.")
    parser.add_argument("label_path", type=str, help="Path to label file.")
    parser.add_argument("--gallery", type=str, help="Path to gallery file.")
    parser.add_argument(
        "--test", action="store_false", help="Whether dataset is train or test."
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--num_worker", type=int, default=0, help="Number of worker.")

    args = parser.parse_args()

    dataset = ImageDataset(
        args.image_dir, args.label_path, train=args.test, gallery_path=args.gallery
    )
    dataloader = DataLoader(
        dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_worker
    )

    data = next(iter(dataloader))
    # data = dataset.get_gallery()
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(data[0].to("cpu")[:64], padding=2, normalize=True).cpu(),
            (1, 2, 0),
        )
    )
    plt.show()
