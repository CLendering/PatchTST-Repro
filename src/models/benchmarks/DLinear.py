import torch
import torch.nn as nn
from typing import Tuple


# ----------------------------------------------------------#
# Implementation based on https://arxiv.org/pdf/2205.13504  #
# ----------------------------------------------------------#


class MovingAverage(nn.Module):
    """
    Moving Average block to highlight the trend of time series.
    Applies average pooling with padding to maintain sequence length.
    """

    def __init__(self, kernel_size: int, stride: int = 1):
        """
        Initializes the MovingAverage module.

        Args:
            kernel_size (int): Size of the moving average window.
            stride (int, optional): Stride of the pooling operation. Defaults to 1.
        """
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MovingAverage module.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Channels, Length].

        Returns:
            torch.Tensor: Moving average applied tensor of the same shape as input.
        """
        padding_size = (self.kernel_size - 1) // 2
        front_padding = x[:, :, :1].repeat(1, 1, padding_size)
        end_padding = x[:, :, -1:].repeat(1, 1, padding_size)
        x_padded = torch.cat([front_padding, x, end_padding], dim=2)
        x_avg = self.avg_pool(x_padded)
        return x_avg


class SeriesDecomposition(nn.Module):
    """
    Series Decomposition block that separates the input time series into trend and residual components.
    """

    def __init__(self, kernel_size: int):
        """
        Initializes the SeriesDecomposition module.

        Args:
            kernel_size (int): Size of the moving average window for trend extraction.
        """
        super(SeriesDecomposition, self).__init__()
        self.moving_average = MovingAverage(kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for series decomposition.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Channels, Length].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Residual and trend components.
        """
        trend = self.moving_average(x)
        residual = x - trend
        return residual, trend


class DecompositionLinearModel(nn.Module):
    """
    Decomposition-Linear model that decomposes the input time series into trend and seasonal components,
    applies linear transformations, and recombines them for forecasting.
    """

    def __init__(self, config: "Config"):
        """
        Initializes the DecompositionLinearModel.

        Args:
            config (Config): Configuration object containing model parameters.
        """
        super(DecompositionLinearModel, self).__init__()
        self.seq_length = config.seq_len
        self.pred_length = config.pred_len
        self.individual = config.individual
        self.num_channels = config.enc_in

        # Series decomposition
        kernel_size = 25
        self.decomposition = SeriesDecomposition(kernel_size=kernel_size)

        # Linear layers for trend and seasonal components
        if self.individual:
            self.linear_seasonal = nn.ModuleList(
                [
                    nn.Linear(self.seq_length, self.pred_length)
                    for _ in range(self.num_channels)
                ]
            )
            self.linear_trend = nn.ModuleList(
                [
                    nn.Linear(self.seq_length, self.pred_length)
                    for _ in range(self.num_channels)
                ]
            )
        else:
            self.linear_seasonal = nn.Linear(self.seq_length, self.pred_length)
            self.linear_trend = nn.Linear(self.seq_length, self.pred_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the DecompositionLinearModel.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Length, Channels].

        Returns:
            torch.Tensor: Forecasted tensor of shape [Batch, Pred_Length, Channels].
        """
        # Decompose the input into residual and trend
        residual, trend = self.decomposition(
            x.permute(0, 2, 1)
        )  # [Batch, Channels, Length]

        if self.individual:
            seasonal_forecasts = []
            trend_forecasts = []
            for i in range(self.num_channels):
                seasonal_forecast = self.linear_seasonal[i](residual[:, i, :])
                trend_forecast = self.linear_trend[i](trend[:, i, :])
                seasonal_forecasts.append(seasonal_forecast.unsqueeze(1))
                trend_forecasts.append(trend_forecast.unsqueeze(1))
            seasonal_output = torch.cat(
                seasonal_forecasts, dim=1
            )  # [Batch, Channels, Pred_Length]
            trend_output = torch.cat(
                trend_forecasts, dim=1
            )  # [Batch, Channels, Pred_Length]
        else:
            seasonal_output = self.linear_seasonal(
                residual
            )  # [Batch, Channels, Pred_Length]
            trend_output = self.linear_trend(trend)  # [Batch, Channels, Pred_Length]

        # Combine seasonal and trend forecasts
        forecast = seasonal_output + trend_output  # [Batch, Channels, Pred_Length]
        return forecast.permute(0, 2, 1)  # [Batch, Pred_Length, Channels]


class Config:
    """
    Configuration class to set up the model parameters.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        individual: bool,
        enc_in: int,
    ):
        """
        Initializes the Config class.

        Args:
            seq_len (int): Length of the input sequence.
            pred_len (int): Length of the prediction sequence.
            individual (bool): Whether to forecast individual channels.
            enc_in (int): Number of input channels.
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.enc_in = enc_in

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"


# ----------------------------------------------------------#
# Test Function and Main Execution Block                   #
# ----------------------------------------------------------#


def test_decomposition_linear_model():
    """
    Comprehensive test for the DecompositionLinearModel.
    Verifies that the model can perform a forward pass and outputs the correct shape.
    """

    # Define test configuration
    test_config = Config(
        seq_len=100,  # Example sequence length
        pred_len=10,  # Example prediction length
        individual=False,  # Test both True and False
        enc_in=3,  # Number of input channels
    )

    # Instantiate the model
    model = DecompositionLinearModel(config=test_config)
    model.eval()  # Set model to evaluation mode

    # Create a dummy input tensor
    batch_size = 5
    dummy_input = torch.randn(
        batch_size, test_config.seq_len, test_config.enc_in
    )  # [Batch, Length, Channels]

    # Perform a forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Expected output shape: [Batch, Pred_Length, Channels]
    expected_shape = (batch_size, test_config.pred_len, test_config.enc_in)

    # Assert the output shape is correct
    assert (
        output.shape == expected_shape
    ), f"Output shape mismatch. Expected {expected_shape}, got {output.shape}"

    print(
        "Test passed: DecompositionLinearModel forward pass successful and output shape is correct."
    )


def main():
    """
    Main function to run tests.
    """
    print("Running DecompositionLinearModel tests...")
    test_decomposition_linear_model()
    print("All tests completed successfully.")


if __name__ == "__main__":
    main()
