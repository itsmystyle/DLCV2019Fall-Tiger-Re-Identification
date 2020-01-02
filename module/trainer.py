import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from sklearn.metrics.pairwise import cosine_distances

from dataset import ImageDataset
from model import ResNet152, SeResNet50, ResNetArcFaceModel, SeResNetArcFaceModel
from metrics import MulticlassAccuracy, Accuracy
from loss import CrossEntropyLabelSmooth
from utils import set_random_seed


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        gallery,
        writer,
        metric,
        val_metric,
        save_dir,
        device,
        accumulate_gradient=1,
    ):
        # prepare model and optimizer
        self.model = model.to(device)
        self.optimizer = optimizer

        print(self.model)

        # prepare loss
        self.criterion = criterion

        # prepare dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.gallery = gallery

        # parameters
        self.accumulate_gradient = accumulate_gradient

        # utils
        self.device = device
        self.writer = writer
        self.metric = metric
        self.val_metric = val_metric
        self.save_dir = save_dir

    def fit(self, epochs):
        print("===> start training ...")
        iters = -1
        best_accuracy = 0.0

        for epoch in range(1, epochs + 1):
            loss, iters = self._run_one_epoch(epoch, iters)
            best_accuracy = self._eval_one_epoch(best_accuracy)

    def _run_one_epoch(self, epoch, iters):
        self.model.train()

        trange = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc="Epoch {}".format(epoch),
        )

        self.metric.reset()
        batch_loss = 0.0

        for idx, batch in trange:
            iters += 1

            images = batch[0].to(self.device)
            labels = batch[1].to(self.device)

            if self.model.__class__.__name__ in [
                "ResNetArcFaceModel",
                "SeResNetArcFaceModel",
            ]:
                preds = self.model(images, labels)
            else:
                preds = self.model(images)

            # calculate loss and update weights
            loss = self.criterion(preds, labels) / self.accumulate_gradient
            if idx % self.accumulate_gradient == 0:
                self.optimizer.zero_grad()
            loss.backward()
            if (idx + 1) % self.accumulate_gradient == 0:
                self.optimizer.step()

            # update metric
            self.metric.update(preds, labels)

            # update loss
            batch_loss += loss.item() * self.accumulate_gradient
            self.writer.add_scalars(
                "Loss",
                {"iter_loss": loss.item(), "avg_loss": batch_loss / (idx + 1)},
                iters,
            )

            # update tqdm
            trange.set_postfix(
                loss=batch_loss / (idx + 1),
                **{self.metric.name: self.metric.print_score()}
            )

        if (idx + 1) % self.accumulate_gradient != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return batch_loss / (idx + 1), iters

    def _eval_one_epoch(self, best_accuracy):
        self.model.eval()

        trange = tqdm(
            enumerate(self.val_loader), total=len(self.val_loader), desc="Valid"
        )

        self.val_metric.reset()

        with torch.no_grad():
            gallery = self.model.extract_features(self.gallery[0].to(self.device))
            gallery = gallery.cpu().numpy()

            gallery_label = self.gallery[1].cpu().numpy()

            for idx, batch in trange:

                images = batch[0].to(self.device)
                labels = batch[1].cpu().numpy()

                feature = self.model.extract_features(images)
                feature = feature.cpu().numpy()

                distance = cosine_distances(feature, gallery)
                min_idx = distance.reshape(-1).argmin()
                preds = gallery_label[min_idx]

                self.val_metric.update(preds, labels)

                # update tqdm
                trange.set_postfix(
                    **{self.val_metric.name: self.val_metric.print_score()}
                )

            # save best acc model
            if self.val_metric.get_score() > best_accuracy:
                print("Best model saved!")
                best_accuracy = self.val_metric.get_score()
                self.save(
                    os.path.join(
                        self.save_dir, "model_best_{:.5f}.pth.tar".format(best_accuracy)
                    )
                )

        return best_accuracy

    def save(self, path):
        torch.save(
            self.model.state_dict(), path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tider re-id net.")
    parser.add_argument("image_dir", type=str, help="Path to image directory.")
    parser.add_argument("label_path", type=str, help="Path to label files.")
    parser.add_argument("query_path", type=str, help="Path to query files.")
    parser.add_argument("gallery_path", type=str, help="Path to gallery files.")
    parser.add_argument("save_dir", type=str, help="Where to save trained model.")
    parser.add_argument("--model", type=str, help="Train which model.")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-6, help="Weight decay rate."
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--n_workers", type=int, default=8, help="Number of worker for dataloader."
    )
    parser.add_argument(
        "--ag",
        type=int,
        default=1,
        help="Accumulate gradients before updating the weight.",
    )
    parser.add_argument(
        "--smoothing", action="store_true", help="Whether to smooth label."
    )
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    set_random_seed(args.random_seed)
    writer = SummaryWriter(os.path.join(args.save_dir, "train_logs"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare dataset
    train_dataset = ImageDataset(args.image_dir, args.label_path)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
    )

    val_dataset = ImageDataset(
        args.image_dir, args.query_path, train=False, gallery_path=args.gallery_path
    )
    val_dataloader = DataLoader(
        val_dataset, shuffle=False, batch_size=1, num_workers=args.n_workers
    )
    gallery_images = val_dataset.get_gallery()

    # prepare model
    if args.model == "resnet152":
        model = ResNet152(train_dataset.get_num_classes())
    elif args.model == "seresnet50":
        model = SeResNet50(train_dataset.get_num_classes(), 512)
    elif args.model == "resnet_arcface":
        model = ResNetArcFaceModel(
            train_dataset.get_num_classes(), 30, 0.5, 512, True, device=device,
        )
    elif args.model == "seresnet_arcface":
        model = SeResNetArcFaceModel(
            train_dataset.get_num_classes(), 30, 0.5, 512, True, device=device,
        )

    # prepare optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # criterion
    if args.smoothing:
        criterion = CrossEntropyLabelSmooth(train_dataset.get_num_classes())
    else:
        criterion = nn.NLLLoss()

    # metric
    metric = MulticlassAccuracy()
    val_metric = Accuracy()

    trainer = Trainer(
        model,
        optimizer,
        criterion,
        train_dataloader,
        val_dataloader,
        gallery_images,
        writer,
        metric,
        val_metric,
        args.save_dir,
        device,
        accumulate_gradient=args.ag,
    )

    trainer.fit(args.epochs)
