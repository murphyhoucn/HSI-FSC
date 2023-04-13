import argparse
from utils import get_dataset
from utils import display_dataset
import visdom

viz = visdom.Visdom()
if not viz.check_connection:
    print("Visdom is not connected. Did you run 'python -m visdom.server' ?")

# Argument parser for CLI interaction
parser = argparse.ArgumentParser(
    description="Run deep learning experiments on" " various hyperspectral datasets"
)
parser.add_argument(
    "--dataset", type=str, default=None, help="Dataset to use."
)
parser.add_argument(
    "--folder",
    type=str,
    help="Folder where to store the "
    "datasets (defaults to the current working directory).",
    default="./Datasets/",
)
args = parser.parse_args()
FOLDER = args.folder
DATASET = args.dataset





img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET, FOLDER)


display_dataset(img, gt, RGB_BANDS, LABEL_VALUES, palette, viz)