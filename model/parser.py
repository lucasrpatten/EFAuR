"""
Parses command line arguments and invokes the train function
"""

import argparse

from train import train


def parse_arguments():
    """Parses command line arguments using argparse

    Returns:
        Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser(description="Simple neural network configuration")

    # Add arguments
    parser.add_argument(
        "--batch_size", type=int, default=24, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="Learning rate for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=150, help="Number of epochs for training"
    )
    parser.add_argument(
        "--activation",
        choices=["relu", "leakyrelu", "swish"],
        default="leakyrelu",
        help="Activation function to use",
    )
    parser.add_argument(
        "--pooling",
        choices=["cls", "mean", "max", "attention"],
        default="attention",
        help="Pooling method to use",
    )

    # Parse arguments
    args = parser.parse_args()

    return args


def invoke_train():
    """Parses command line arguments and invokes the train function
    """
    # Parse the arguments
    args = parse_arguments()
    train(args.batch_size, args.epochs, args.learning_rate, args.pooling, args.activation)

if __name__ == "__main__":
    invoke_train()
