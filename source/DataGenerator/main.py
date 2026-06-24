import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dataset import *


def main():
    generate_dataset_no_augmentation_new()
    # generate_dataset()
    # balance_dataset()
    # split_dataset()
    # create_data_yaml()
    # generate_eval_datasets()

if __name__ == "__main__":
    main()