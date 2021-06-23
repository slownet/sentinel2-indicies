"""
Utilities used by example notebooks
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray


def plot_image(image, factor=1.0, clip_range=None, **kwargs):
    """
    Utility function for plotting RGB images.
    """

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)

    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()


def save_image(image: ndarray, factor=1.0, clip_range=None, img_path=None, **kwargs):
    """
    Utility function for saving RGB images.
    """

    width, height = image.shape

    plt.figure(figsize=(width / 100, height / 100))

    ax = plt.gca()

    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)

    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(img_path, dpi=100)
