"""
Data Generation Pipeline for EFAuR

Author: Lucas Patten
"""

import argparse
import logging

from get_data import by_author, mirror_data
from get_metadata import extract_metadata, get_tarball
from generate_dataset import generate_dataset


def argument_parsing():
    """Parses command line arguments using argparse"""
    parser = argparse.ArgumentParser(
        prog="pipeline.py", description="Data Generation Pipeline for EFAuR"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/lucasrp/compute/gutenberg",
        help="The base path of the directory in which to store the data",
    )

    parser.add_argument(
        "--verbosity",
        "-v",
        dest="verbosity",
        type=int,
        choices=[0, 1, 2, 3, 4, 5],
        default=3,
        help="Verbosity level",
    )

    parser.add_argument(
        "--threads",
        dest="thread_count",
        type=int,
        default=16,
        help="Number of threads to use (only applicable for --generate-dataset)",
    )

    parser.add_argument("--dataset_size", type=int, default=100000, help="Dataset size")
    parser.add_argument(
        "--train_percent", type=float, default=0.8, help="Training percent"
    )
    parser.add_argument(
        "--val_percent", type=float, default=0.1, help="Validation percent"
    )
    parser.add_argument("--test_percent", type=float, default=0.1, help="Test percent")

    actions = parser.add_mutually_exclusive_group()

    actions.add_argument(
        "--download", action="store_true", help="Download data (requires internet)"
    )
    actions.add_argument("--process", action="store_true", help="Process data")
    actions.add_argument("--mirror-data", action="store_true", help="Mirror data")
    actions.add_argument("--get-tarball", action="store_true", help="Get tarball")
    actions.add_argument("--extract-metadata", action="store_true", help="Extract metadata")
    actions.add_argument("--by-author", action="store_true", help="By author")
    actions.add_argument(
        "--generate-dataset", action="store_true", help="Generate dataset"
    )
    actions.add_argument(
        "--offline-only",
        action="store_true",
        help="All offline functions (get_metadata, by_author, generate_dataset)",
    )
    actions.add_argument(
        "--all", action="store_true", help="Complete Pipeline (all actions)"
    )

    args = parser.parse_args()

    verbosity = 50 - args.verbosity * 10
    logging.basicConfig(
        level=verbosity, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if args.download:
        mirror_data(args.data_dir)
        get_tarball()
    if args.process:
        extract_metadata(args.data_dir)
        by_author(args.data_dir)
    if args.mirror_data:
        mirror_data(args.data_dir)
    if args.get_tarball:
        get_tarball()
    if args.extract_metadata:
        extract_metadata(args.data_dir)
    if args.by_author:
        by_author(args.data_dir)
    if args.generate_dataset:
        generate_dataset(
            args.data_dir,
            dataset_size=args.dataset_size,
            train_percent=args.train_percent,
            val_percent=args.val_percent,
            test_percent=args.test_percent,
            num_processes=args.thread_count,
        )
    if args.offline_only:
        extract_metadata(args.data_dir)
        by_author(args.data_dir)
        generate_dataset(
            args.data_dir,
            dataset_size=args.dataset_size,
            train_percent=args.train_percent,
            val_percent=args.val_percent,
            test_percent=args.test_percent,
            num_processes=args.thread_count,
        )
    if args.all:
        mirror_data(args.data_dir)
        get_tarball()
        extract_metadata(args.data_dir)
        by_author(args.data_dir)
        generate_dataset(
            args.data_dir,
            dataset_size=args.dataset_size,
            train_percent=args.train_percent,
            val_percent=args.val_percent,
            test_percent=args.test_percent,
            num_processes=args.thread_count,
        )


if __name__ == "__main__":
    argument_parsing()
