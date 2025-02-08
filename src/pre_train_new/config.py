import argparse
from dataclasses import dataclass
from typing import Union


@dataclass
class PreTrainingConfig:
    # General parameters
    model_identifier: str
    seed: int
    model: str
    input_length: int
    prediction_length: int
    dataset: str
    checkpoint_dir: str
    features: str

    # Formers
    d_model: int
    n_heads: int
    num_encoder_layers: int
    d_fcn: int
    dropout: float



    # PatchTST model parameters
    kernel_size: int
    patch_length: int
    stride: int
    patch_padding: str
    head_dropout: float
    revin: bool

    # Training parameters
    batch_size: int
    epochs: int
    learning_rate: float
    patience: int
    num_workers: int
    use_cuda: bool


    # Pretraining parameters
    mask_ratio: float


def get_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for PatchTST training.
    """
    parser = argparse.ArgumentParser(description="PatchTST few-shot learning")

    # General parameters
    parser.add_argument(
        "--model_identifier",
        type=str,
        default="PatchTST",
        help="Model identifier",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model",
        type=str,
        default="PatchTST",
        help="Model architecture (default: PatchTST)",
    )
    parser.add_argument(
        "--input_length",
        type=int,
        default=512,
        help="Input sequence length (default: 512)",
    )
    parser.add_argument(
        "--prediction_length",
        type=int,
        default=96,
        help="Prediction length (default: 96)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="etth1",
        help="Dataset name (default: etth1)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="Forecasting Task - [M, S, MS]; M: Multivariate Predict Multivariate, S: Univariate Predict Univariate, MS: Multivariate Predict Univariate",
    )


    parser.add_argument(
        "--d_model",
        type=int,
        default=128,
        help="Model dimension (default: 128)",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=16,
        help="Number of heads (default: 16)",
    )
    parser.add_argument(
        "--num_encoder_layers",
        type=int,
        default=3,
        help="Number of encoder layers (default: 2)",
    )
    parser.add_argument(
        "--d_fcn",
        type=int,
        default=512,
        help="Fully connected layer dimension (default: 512)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate (default: 0.2)",
    )


    # PatchTST model parameters
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=25,
        help="Kernel size (default: 25)",
    )
    parser.add_argument(
        "--patch_length",
        type=int,
        default=12,
        help="Patch length (default: 12)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=12,
        help="Stride (default: 12)",
    )
    parser.add_argument(
        "--patch_padding",
        type=str,
        default="end",
        help="'None: None; end: Padding on the end",
    )
    parser.add_argument(
        "--head_dropout",
        type=float,
        default=0.2,
        help="Head dropout (default: 0.2)",
    )
    parser.add_argument(
        "--revin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable RevIn",
    )
    # Training parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs (default: 100)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Patience for early stopping (default: 10)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for data loading (default: 8)",
    )
    parser.add_argument(
        "--use_cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CUDA",
    )
    


    # Pretraining parameters
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.4,
        help="Masking ratio for the input (default)"
    )
    return parser


def parse_args() -> PreTrainingConfig:
    """
    Parse command-line arguments and return a configuration object.
    """
    parser = get_parser()
    args = parser.parse_args()

    return PreTrainingConfig(
        model_identifier=args.model_identifier,
        seed=args.seed,
        model=args.model,
        input_length=args.input_length,
        prediction_length=args.prediction_length,
        dataset=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        features=args.features,
        kernel_size=args.kernel_size,
        patch_length=args.patch_length,
        stride=args.stride,
        patch_padding=args.patch_padding,
        head_dropout=args.head_dropout,
        revin=args.revin,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        num_workers=args.num_workers,
        use_cuda=args.use_cuda,

        d_model=args.d_model,
        n_heads=args.n_heads,
        num_encoder_layers=args.num_encoder_layers,
        d_fcn=args.d_fcn,
        dropout=args.dropout,

        mask_ratio=args.mask_ratio,

    )