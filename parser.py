from __future__ import absolute_import
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description="HW4")

    # Datasets parameters
    parser.add_argument(
        "--image_dir",
        type=str,
        default="final_data/imgs/",
        help="path to trimmed video data directory",
    )
    parser.add_argument(
        "--label_path",
        type=str,
        default="final_data/train.csv",
        help="path to trimmed video label directory",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="pretrained_model/se_resnet101.pth",
        help="path to pretrained imagenet model",
    )

    """ training parameters """
    parser.add_argument(
        "--epochs", default=100, type=int, help="num of validation iterations"
    )
    parser.add_argument(
        "--num_worker", default=1, type=int, help="num of workers to load data"
    )
    parser.add_argument("--train_batch", default=4, type=int, help="train batch size")
    parser.add_argument("--test_batch", default=4, type=int, help="test batch size")
    parser.add_argument(
        "--lr", default=0.0002, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--gamma", default=0.9, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--plot_task", default=1, type=int, help="plot tsne of which task"
    )

    # others
    parser.add_argument(
        "--log_dir",
        type=str,
        default="log/",
        help="Record train/val/test acc, loss or something like that",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_dir/",
        help="Output the result of model",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="model_log/",
        help="Directory for saving trained model",
    )
    parser.add_argument(
        "--fig_dir", type=str, default="fig/", help="Directory for saving figure"
    )

    # Testing
    parser.add_argument("--test", action="store_true", help="whether test mode or not")
    parser.add_argument(
        "--test_model_path",
        type=str,
        default="model_log/task3/task3_valid_0_0.505.chkpt",
        help="Path to testing model",
    )

    args = parser.parse_args()

    return args
