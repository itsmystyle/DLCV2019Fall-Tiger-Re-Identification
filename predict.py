import argparse

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
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
from module.metrics.utils import re_ranking
from module.utils import set_random_seed

NUM_CLASSES = 72

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference tiger re-id net.")
    parser.add_argument("image_dir", type=str, help="Path to image directory.")
    parser.add_argument("query_path", type=str, help="Path to query files.")
    parser.add_argument("gallery_path", type=str, help="Path to gallery files.")
    parser.add_argument("model_dir", type=str, help="Where to save trained model.")
    parser.add_argument(
        "model_architecture", type=str, help="Which model architecture to use."
    )
    parser.add_argument("output_dir", type=str, help="Output file path.")
    parser.add_argument(
        "--n_workers", type=int, default=8, help="Number of worker for dataloader."
    )
    parser.add_argument("--scale", type=int, default=30, help="Scale for arcface.")
    parser.add_argument("--margin", type=float, default=0.5, help="Margin for arcface.")
    parser.add_argument(
        "--feature_dim", type=int, default=512, help="Final features dimension."
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare dataset
    dataset = ImageDataset(
        args.image_dir,
        args.query_path,
        train=False,
        gallery_path=args.gallery_path,
        with_label=False,
    )
    dataloader = DataLoader(
        dataset, shuffle=False, batch_size=1, num_workers=args.n_workers
    )
    gallery_images = dataset.get_gallery()

    # prepare model
    if args.model_architecture == "resnet152":
        model = ResNet152(NUM_CLASSES)
    elif args.model_architecture == "seresnet50":
        model = SeResNet50(NUM_CLASSES, args.feature_dim)
    elif args.model_architecture == "seresnet152":
        model = SeResNet152(NUM_CLASSES, args.feature_dim)
    elif args.model_architecture == "seresnext50":
        model = SeResNeXt50(NUM_CLASSES, args.feature_dim)
    elif args.model_architecture == "resnet_arcface":
        model = ResNetArcFaceModel(
            NUM_CLASSES, args.scale, args.margin, args.feature_dim, True, device=device,
        )
    elif args.model_architecture == "seresnet_arcface":
        model = SeResNetArcFaceModel(
            NUM_CLASSES, args.scale, args.margin, args.feature_dim, True, device=device,
        )
    elif args.model_architecture == "seresnext50_arcface":
        model = SeResNeXtArcFaceModel(
            NUM_CLASSES,
            args.scale,
            args.margin,
            args.feature_dim,
            50,
            True,
            device=device,
        )
    elif args.model_architecture == "seresnext101_arcface":
        model = SeResNeXtArcFaceModel(
            NUM_CLASSES,
            args.scale,
            args.margin,
            args.feature_dim,
            101,
            True,
            device=device,
        )
    elif args.model_architecture == "nasnet":
        model = NASNet(NUM_CLASSES, args.feature_dim)

    # load model
    model.load_state_dict(torch.load(args.model_dir))
    model.to(device)
    model.eval()

    # get gallery features
    trange = tqdm(enumerate(dataloader), total=len(dataloader), desc="Inferencing")

    with torch.no_grad():
        gallery = model.extract_features(gallery_images["images"].to(device))
        gallery = gallery.cpu().numpy()
        gallery_path = gallery_images["img_paths"]

        feats = []
        labels = []

        for idx, batch in trange:

            images = batch["images"].to(device)

            feature = model.extract_features(images)
            feature = feature.cpu().numpy()

            if args.re_ranking:
                feats.append(feature)
            else:
                distance = cosine_distances(feature, gallery)
                min_idx = distance.reshape(-1).argmin()
                preds = gallery_path[min_idx]

                labels.append(preds)

    if args.re_ranking:
        feats = np.concatenate(feats)
        distmat = re_ranking(
            feats, gallery, k1=args.k1, k2=args.k2, lambda_value=args.lambda_value
        )
        labels = gallery_path[distmat.argmin(axis=1)]
    else:
        labels = np.asarray(labels)

    with open(args.output_dir, "w") as fout:
        for label in labels:
            fout.write("{}\n".format(label))
