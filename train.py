import os
import test
import pdb
import glob
import random
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as tvmodels

""" own package """
import parser
from module.dataset import ImageDataset
from lr_scheduler import WarmupMultiStepLR
from tigerNet import TigerNet


"""  setup random seed and device """
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def train(args):
    train_dataset = ImageDataset(args.image_dir, args.label_path, train=True)
    dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=args.train_batch,
        num_workers=args.num_worker,
    )

    """ Define model """
    # num_classes = 72
    model = TigerNet(
        72,
        1,  # Last stride of backbone
        "",
        "bnneck",
        "after",
        "se_resnet101",
        "imagenet",
    ).to(device)

    """ Define Loss Function """
    criterion = nn.NLLLoss().to(device)

    """ Define optimizer & scheduler """
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = 3e-4
        weight_decay = 0.0005
        if "bias" in key:
            lr = 0.0005 * 2
            weight_decay = 0.0
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = getattr(torch.optim, "Adam")(params)

    scheduler = WarmupMultiStepLR(optimizer, (30, 55), 0.1, 1.0 / 3, 500, "linear")

    """ Start Training """
    trange = tqdm(enumerate(dataloader), total=len(dataloader), desc="Tiger Training")

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        total_loss = 0
        n_tiger_total = 0
        n_correct_total = 0

        for idx, (images, labels) in trange:
            images, labels = images.to(device), labels.to(device)
            pred, global_feat = model(images, True)
            loss = criterion(pred, labels)
            loss.backward()
            pdb.set_trace()

            if idx % 4 == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            pred = pred.max(1)[1]
            labels = labels.contiguous().view(-1)
            n_correct = pred.eq(labels).sum().item()
            n_tiger = labels.shape[0]
            n_tiger_total += n_tiger
            n_correct_total += n_correct
            acc = n_correct_total / n_tiger_total

            trange.set_postfix(
                {
                    "epoch": "{}".format(epoch),
                    "loss": "{0:.4f}".format(total_loss / (idx + 1)),
                    "acc": "{0:.3f}".format(acc),
                }
            )
        
        trange = tqdm(enumerate(valid_dataloader), total=len(valid_dataloader), desc="Task1 Validating")
        total_loss = 0
        n_video_total = 0
        n_correct_total = 0
        acc = 0
        with torch.no_grad():
            model.eval()
            for idx, (imgs, labels) in trange:
                # imgs = [batch, 240, 320, 3]
                # labels = [batch] 
                imgs = imgs.to(device)
                pred = model(imgs)
                labels = labels.to(device)

                pred = pred.max(1)[1]
                labels = labels.contiguous().view(-1)
                n_correct = pred.eq(labels).sum().item()
                n_video = labels.shape[0]
                n_video_total += n_video
                n_correct_total += n_correct
                acc = n_correct_total / n_video_total

                trange.set_postfix({"epoch":"{}".format(epoch), "acc":"{0:.3f}".format(acc)})

            if acc > best_acc:
                best_acc = acc
                save_path = os.path.join(args.model_dir, 'task1_valid_{}_{:.3f}.chkpt'.format(epoch, acc))
                save_model(model, save_path)


        scheduler.step()



if __name__ == "__main__":
    args = parser.arg_parse()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    train(args)
