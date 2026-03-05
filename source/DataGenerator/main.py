import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dataset import generate_dataset, split_dataset, create_data_yaml


def main():
    generate_dataset()
    split_dataset()
    create_data_yaml()


if __name__ == "__main__":
    main()