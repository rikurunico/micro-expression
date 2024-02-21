from typing import Literal, Tuple
import os, cv2
import numpy as np


def extract_component_as_image(
    image,
    frameNumber,
    objectRectangle: Tuple[int, int, int, int],
    objectName: Literal[
        "mouth",
        "eye_left",
        "eye_right",
        "eyebrow_left",
        "eyebrow_right",
        "nose_right",
        "nose_left",
    ],
):
    # Construct directory path for saving images
    file_dir = f"dataset/element_images/{objectName}/block-{frameNumber:02}"

    # Create directory if it doesn't exist
    os.makedirs(file_dir, exist_ok=True)

    y_top, x_right, y_bottom, x_left = objectRectangle
    selected_component_image = image[y_top:y_bottom, x_left:x_right]

    # Save cropped component image
    cv2.imwrite(
        os.path.join(file_dir, f"frame-{frameNumber:02}.jpg"), selected_component_image
    )

    # Split the component image into 5x5 blocks
    blocks = split_into_blocks(selected_component_image, block_size=(5, 5))

    # Save each block as a separate image
    for i, block in enumerate(blocks):
        block_filename = f"block_{i:02}.jpg"
        cv2.imwrite(os.path.join(file_dir, block_filename), block)


def split_into_blocks(image, block_size=(5, 5)):
    """Splits an image into blocks of the specified size."""
    h, w, c = image.shape
    block_height, block_width = block_size
    num_blocks_h = h // block_height
    num_blocks_w = w // block_width

    blocks = np.zeros(
        (num_blocks_h * num_blocks_w, block_height, block_width, c), dtype=image.dtype
    )
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            block_start_h = i * block_height
            block_end_h = (i + 1) * block_height
            block_start_w = j * block_width
            block_end_w = (j + 1) * block_width
            blocks[i * num_blocks_w + j] = image[
                block_start_h:block_end_h, block_start_w:block_end_w
            ]

    return blocks
