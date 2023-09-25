import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))

        self.model = nn.Sequential(
          # First conv + maxpool + relu
          nn.Conv2d(3, 8, 3, padding=1),
          nn.BatchNorm2d(8),
          nn.MaxPool2d(2, 2),
          nn.ReLU(),
          # nn.Dropout2d(dropout),

          # Second conv + maxpool + relu
          nn.Conv2d(8, 16, 3, padding=1),
          nn.BatchNorm2d(16),
          nn.MaxPool2d(2, 2),
          nn.ReLU(),
          # nn.Dropout2d(dropout),

          # # # Third conv + maxpool + relu
          nn.Conv2d(16, 32, 3, padding=1),
          nn.BatchNorm2d(32),
          nn.MaxPool2d(2, 2),
          nn.ReLU(),

          # Flatten feature maps
          nn.Flatten(),

          # Fully connected layers. This assumes
          # that the input image was 256x256 round((96800 + num_classes)/2)
          nn.Linear(25088, 256),
          nn.BatchNorm1d(256),
          nn.Dropout(dropout),
          nn.ReLU(),
          nn.Linear(256, num_classes)
    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
      
      # nn.Sequential will call the layers
      x =  self.model(x)
      return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
