import datetime
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

from usbmd import log


def translate(array, range_from, range_to):
    """Map values in array from one range to other.

    Args:
        array (ndarray): input array.
        range_from (Tuple): lower and upper bound of original array.
        range_to (Tuple): lower and upper bound to which array should be mapped.

    Returns:
        (ndarray): translated array
    """
    left_min, left_max = range_from
    right_min, right_max = range_to

    # Convert the left range into a 0-1 range (float)
    value_scaled = (array - left_min) / (left_max - left_min)

    # Convert the 0-1 range into a value in the right range.
    return right_min + (value_scaled * (right_max - right_min))


def get_date_string(string: str = None):
    """Generate a date string for current time, according to format specified by
    `string`. Refer to the documentation of the datetime module for more information
    on the formatting options.

    If no string is specified, the default format is used: "%Y_%m_%d_%H%M%S".
    """
    if string is not None and not isinstance(string, str):
        raise TypeError("Input must be a string.")

    # Get the current time
    now = datetime.datetime.now()

    # If no string is specified, use the default format
    if string is None:
        string = "%Y_%m_%d_%H%M%S"

    # Generate the date string
    date_str = now.strftime(string)

    return date_str


def strtobool(val: str):
    """Convert a string representation of truth to True or False.

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    assert isinstance(val, str), f"Input value must be a string, not {type(val)}"
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError(f"invalid truth value {val}")


def to_8bit(image, dynamic_range: Union[None, tuple] = None, pillow: bool = True):
    """Convert image to 8 bit image [0, 255]. Clip between dynamic range.

    Args:
        image (ndarray): Input image(s). Should be in between dynamic range.
        dynamic_range (tuple, optional): Dynamic range of input image(s).
        pillow (bool, optional): Whether to return PIL image. Defaults to True.

    Returns:
        image (ndarray): Output 8 bit image(s) [0, 255].

    """
    if dynamic_range is None:
        dynamic_range = (-60, 0)

    image = image.numpy()

    image = np.clip(image, *dynamic_range)
    image = translate(image, dynamic_range, (0, 255))
    image = image.astype(np.uint8)
    if pillow:
        image = Image.fromarray(image)
    return image


def _assert_uint8_images(images: np.ndarray):
    """
    Asserts that the input images have the correct properties.

    Args:
        images (np.ndarray): The input images.

    Raises:
        AssertionError: If the dtype of images is not uint8.
        AssertionError: If the shape of images is not (n_frames, height, width, channels)
            or (n_frames, height, width) for grayscale images.
        AssertionError: If images have anything other than 1 (grayscale),
            3 (rgb) or 4 (rgba) channels.
    """
    assert (
        images.dtype == np.uint8
    ), f"dtype of images should be uint8, got {images.dtype}"

    assert images.ndim in (3, 4), (
        "images must have shape (n_frames, height, width, channels),"
        f" or (n_frames, height, width) for grayscale images. Got {images.shape}"
    )

    if images.ndim == 4:
        assert images.shape[-1] in (1, 3, 4), (
            "Grayscale images must have 1 channel, "
            "RGB images must have 3 channels, and RGBA images must have 4 channels. "
            f"Got shape: {images.shape}, channels: {images.shape[-1]}"
        )


def grayscale_to_rgb(image):
    """Converts a grayscale image to an RGB image.

    Args:
        image (ndarray): Grayscale image. Must have shape (height, width).

    Returns:
        ndarray: RGB image.
    """
    assert image.ndim == 2, "Input image must be grayscale."
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def preprocess_for_saving(images):
    """Preprocesses images for saving to GIF or MP4.

    Args:
        images (ndarray, list[ndarray]): Images. Must have shape (n_frames, height, width, channels)
            or (n_frames, height, width).
    """
    images = np.array(images)
    _assert_uint8_images(images)

    # Remove channel axis if it is 1 (grayscale image)
    if images.ndim == 4 and images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)

    # convert grayscale images to RGB
    if images.ndim == 3:
        images = [grayscale_to_rgb(image) for image in images]
        images = np.array(images)

    return images


def save_to_gif(images, filename, fps=20, shared_color_palette=False):
    """Saves a sequence of images to .gif file.
    Args:
        images: list of images (numpy arrays). Must have shape
            (n_frames, height, width, channels) or (n_frames, height, width).
            If channel axis is not present, or is 1, grayscale image is assumed,
            which is then converted to RGB. Images should be uint8.
        filename: string containing filename to which data should be written.
        fps: frames per second of rendered format.
        shared_color_palette (bool, optional): if True, creates a global
            color palette across all frames, ensuring consistent colors
            throughout the GIF. Defaults to False, which is default behavior
            of PIL.Image.save.
            Note: `True` can cause slow saving for longer sequences.
    """
    assert isinstance(
        filename, (str, Path)
    ), f"Filename must be a string or Path object, not {type(filename)}"
    images = preprocess_for_saving(images)

    if fps > 50:
        log.warning(f"Cannot set fps ({fps}) > 50. Setting it automatically to 50.")
        fps = 50

    duration = 1 / (fps) * 1000  # milliseconds per frame

    pillow_imgs = [Image.fromarray(img) for img in images]

    if shared_color_palette:
        # Apply the same palette to all frames without dithering for consistent color mapping
        # Convert all images to RGB and combine their colors for palette generation
        all_colors = np.vstack(
            [np.array(img.convert("RGB")).reshape(-1, 3) for img in pillow_imgs]
        )
        combined_image = Image.fromarray(all_colors.reshape(-1, 1, 3))

        # Generate palette from all frames
        global_palette = combined_image.quantize(
            colors=256,
            method=Image.MEDIANCUT,  # pylint: disable=no-member
            kmeans=1,
        )

        # Apply the same palette to all frames without dithering
        pillow_imgs = [
            img.convert("RGB").quantize(
                palette=global_palette,
                dither=Image.NONE,  # pylint: disable=no-member
            )
            for img in pillow_imgs
        ]

    pillow_img, *pillow_imgs = pillow_imgs

    pillow_img.save(
        fp=filename,
        format="GIF",
        append_images=pillow_imgs,
        save_all=True,
        loop=0,
        duration=duration,
        interlace=False,
        optimize=False,
    )
    return log.success(f"Succesfully saved GIF to -> {log.yellow(filename)}")
