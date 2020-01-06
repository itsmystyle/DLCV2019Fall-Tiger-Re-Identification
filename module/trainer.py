import os

import torch
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_distances


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
