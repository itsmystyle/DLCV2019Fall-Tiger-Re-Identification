import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from sklearn.metrics.pairwise import cosine_distances

from module.dataset import ImageDataset
from module.model import (
    ResNet152,
    SeResNet50,
    SeResNet152,
    SeResNeXt50,
    ResNetArcFaceModel,
    SeResNetArcFaceModel,
    SeResNeXtArcFaceModel,
    NASNet,
)
from module.metrics.metrics import MulticlassAccuracy, Accuracy, ReRankingAccuracy
from module.loss import CrossEntropyLabelSmooth, TripletLoss, CenterLoss
from module.lr_scheduler import WarmupMultiStepLR
from module.utils import set_random_seed


# decay rate of learning rate
GAMMA = 0.1
# decay step of learning rate
STEPS = (10, 35)

# warm up factor
WARMUP_FACTOR = 1.0 / 3
# iterations of warm up
WARMUP_ITERS = 20
# method of warm up, option: 'constant','linear'
WARMUP_METHOD = "linear"


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        gallery,
        scheduler,
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

        # scheduler
        self.scheduler = scheduler

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

            if self.scheduler:
                self.scheduler.step()

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

            images = batch["images"].to(self.device)
            labels = batch["labels"].to(self.device)

            if self.model.__class__.__name__ in [
                "ResNetArcFaceModel",
                "SeResNetArcFaceModel",
                "SeResNeXtArcFaceModel",
            ]:
                preds, features = self.model(images, labels)
            else:
                preds, features = self.model(images)

            # calculate loss and update weights
            loss = 0.0
            for criterion in self.criterion:
                name = criterion[0]
                scale = criterion[2]
                criterion = criterion[1]

                if name in ["TripletLoss", "CenterLoss"]:
                    loss += scale * criterion(features, labels)
                else:
                    loss += scale * criterion(preds, labels)

            loss /= self.accumulate_gradient
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
            gallery = self.model.extract_features(
                self.gallery["images"].to(self.device)
            )
            gallery = gallery.cpu().numpy()

            gallery_label = self.gallery["labels"].cpu().numpy()

            if self.val_metric.name == "Re-Rank 1":
                self.val_metric.update(gallery, gallery_label)

            for idx, batch in trange:

                images = batch["images"].to(self.device)
                labels = batch["labels"].cpu().numpy()

                feature = self.model.extract_features(images)
                feature = feature.cpu().numpy()

                if self.val_metric.name == "Re-Rank 1":
                    self.val_metric.update(feature, labels)

                    # update tqdm
                    trange.set_postfix(**{self.val_metric.name: "TBD"})
                else:
                    distance = cosine_distances(feature, gallery)
                    min_idx = distance.reshape(-1).argmin()
                    preds = gallery_label[min_idx]

                    self.val_metric.update(preds, labels)

                    # update tqdm
                    trange.set_postfix(
                        **{self.val_metric.name: self.val_metric.print_score()}
                    )

            _val_score = self.val_metric.get_score()

            print("Validation Rank-1 score: {:.5f}".format(_val_score))

            # save best acc model
            if _val_score > best_accuracy:
                print("Best model saved!")
                best_accuracy = _val_score
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
    parser = argparse.ArgumentParser(description="Train tiger re-id net.")
    parser.add_argument("image_dir", type=str, help="Path to image directory.")
    parser.add_argument("label_path", type=str, help="Path to label files.")
    parser.add_argument("query_path", type=str, help="Path to query files.")
    parser.add_argument("gallery_path", type=str, help="Path to gallery files.")
    parser.add_argument("save_dir", type=str, help="Where to save trained model.")
    parser.add_argument(
        "--model", type=str, default="resnet152", help="Train which model."
    )
    parser.add_argument("--epochs", type=int, default=5, help="Epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-6, help="Weight decay rate."
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--scale", type=float, default=30.0, help="Scale for arcface.")
    parser.add_argument("--margin", type=float, default=0.5, help="Margin for arcface.")
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
    parser.add_argument(
        "--triplet", action="store_true", help="Whether to use triplet loss."
    )
    parser.add_argument(
        "--center", action="store_true", help="Whether to use center loss."
    )
    parser.add_argument(
        "--dist",
        type=str,
        default="euclidean",
        help="Which distrance function to user, euclidean or cos.",
    )
    parser.add_argument(
        "--feature_dim", type=int, default=512, help="Final features dimension."
    )
    parser.add_argument(
        "--lr_scheduler", action="store_true", help="Whether to use scheduler."
    )
    parser.add_argument(
        "--re_ranking",
        action="store_true",
        help="Whether to use re-ranking to calculate rank 1.",
    )
    parser.add_argument("--k1", type=int, default=20, help="K1 value for re-ranking.")
    parser.add_argument("--k2", type=int, default=6, help="K2 value for re-ranking.")
    parser.add_argument(
        "--lambda_value", type=float, default=0.3, help="Lambda value for re-ranking."
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
        model = SeResNet50(train_dataset.get_num_classes(), args.feature_dim)
    elif args.model == "seresnet152":
        model = SeResNet152(train_dataset.get_num_classes(), args.feature_dim)
    elif args.model == "seresnext50":
        model = SeResNeXt50(train_dataset.get_num_classes(), args.feature_dim)
    elif args.model == "resnet_arcface":
        model = ResNetArcFaceModel(
            train_dataset.get_num_classes(),
            args.scale,
            args.margin,
            args.feature_dim,
            True,
            device=device,
        )
    elif args.model == "seresnet_arcface":
        model = SeResNetArcFaceModel(
            train_dataset.get_num_classes(),
            args.scale,
            args.margin,
            args.feature_dim,
            True,
            device=device,
        )
    elif args.model == "seresnext50_arcface":
        model = SeResNeXtArcFaceModel(
            train_dataset.get_num_classes(),
            args.scale,
            args.margin,
            args.feature_dim,
            50,
            True,
            device=device,
        )
    elif args.model == "seresnext101_arcface":
        model = SeResNeXtArcFaceModel(
            train_dataset.get_num_classes(),
            args.scale,
            args.margin,
            args.feature_dim,
            101,
            True,
            device=device,
        )
    elif args.model == "nasnet":
        model = NASNet(train_dataset.get_num_classes(), args.feature_dim)

    # prepare optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # criterion
    criterion = []
    if args.smoothing:
        criterion += [
            (
                "CELoss",
                CrossEntropyLabelSmooth(train_dataset.get_num_classes(), device=device),
                1.0,
            )
        ]
    else:
        criterion += [("CELoss", nn.NLLLoss(), 1.0)]

    if args.triplet:
        criterion += [("TripletLoss", TripletLoss(0.3, args.dist), 4.0)]

    if args.center:
        criterion += [
            (
                "CenterLoss",
                CenterLoss(
                    num_classes=train_dataset.get_num_classes(),
                    feat_dim=args.feature_dim,
                    device=device,
                ),
                1.0,
            )
        ]

    if args.model == "":
        criterion += [("ReconstructionLoss", nn.MSELoss(), 1.0)]

    # scheduler
    scheduler = None
    if args.lr_scheduler:
        scheduler = WarmupMultiStepLR(
            optimizer, STEPS, GAMMA, WARMUP_FACTOR, WARMUP_ITERS, WARMUP_METHOD,
        )

    # metric
    metric = MulticlassAccuracy()
    if args.re_ranking:
        val_metric = ReRankingAccuracy(
            num_query=len(val_dataset),
            max_rank=gallery_images["labels"].shape[0],
            k1=args.k1,
            k2=args.k2,
            lambda_value=args.lambda_value,
        )
    else:
        val_metric = Accuracy()

    trainer = Trainer(
        model,
        optimizer,
        criterion,
        train_dataloader,
        val_dataloader,
        gallery_images,
        scheduler,
        writer,
        metric,
        val_metric,
        args.save_dir,
        device,
        accumulate_gradient=args.ag,
    )

    trainer.fit(args.epochs)
