import torch
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """
    Loads the MNIST dataset using PyTorch's torchvision.

    Parameters
    ----------
    normalize : bool
        Whether to scale the image pixel values to [0, 1].
    flatten : bool
        Whether to flatten the 28x28 images to 784-dimensional vectors.
    one_hot_label : bool
        Whether to convert labels to one-hot encoding.

    Returns
    -------
    (train_x, train_y), (test_x, test_y) : Tuple of numpy arrays
    """
    transform_list = [transforms.ToTensor()]  # ToTensor scales pixels to [0, 1]
    transform = transforms.Compose(transform_list)

    # Load datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Convert images and labels to NumPy arrays
    train_x = train_dataset.data.numpy()
    train_y = train_dataset.targets.numpy()
    test_x = test_dataset.data.numpy()
    test_y = test_dataset.targets.numpy()

    # Normalize if requested
    if normalize:
        train_x = train_x.astype("float32") / 255.0
        test_x = test_x.astype("float32") / 255.0

    # Flatten if requested
    if flatten:
        train_x = train_x.reshape(-1, 28 * 28)
        test_x = test_x.reshape(-1, 28 * 28)
    else:
        train_x = train_x.reshape(-1, 1, 28, 28)
        test_x = test_x.reshape(-1, 1, 28, 28)

    # One-hot encode labels if requested
    if one_hot_label:
        train_y = np.eye(10)[train_y]
        test_y = np.eye(10)[test_y]

    return (train_x, train_y), (test_x, test_y)

def _change_one_hot_label(labels):
    """
    Converts label array to one-hot encoded format.

    Parameters
    ----------
    labels : array of int

    Returns
    -------
    one_hot_labels : array of shape (n_samples, 10)
    """
    return np.eye(10)[labels]

def img_show(img):
    """
    Displays a single image (flattened or 2D array).

    Parameters
    ----------
    img : numpy array of shape (784,) or (28, 28)
    """
    if img.ndim == 1:
        img = img.reshape(28, 28)
    pil_img = Image.fromarray(np.uint8(img * 255))  # Assume input is normalized
    pil_img.show()

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784
(train_x, train_y), (test_x, test_y) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
