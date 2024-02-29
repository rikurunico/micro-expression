import os
import cv2
import numpy as np
from typing import Tuple, Literal


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
    selected_component_image = image.copy()

    # Draw rectangle around the selected component
    cv2.rectangle(
        selected_component_image,
        (x_left, y_top),
        (x_right, y_bottom),
        (0, 255, 0),
        1,
    )

    # Crop the selected component
    selected_component_image = selected_component_image[y_top:y_bottom, x_left:x_right]

    # grayscale the image
    selected_component_image = cv2.cvtColor(
        selected_component_image, cv2.COLOR_BGR2GRAY
    )

    # Save the annotated image
    cv2.imwrite(
        os.path.join(file_dir, f"annotated_frame-{frameNumber:02}.jpg"),
        selected_component_image,
    )

    # Save the whole image with block lines
    split_into_blocks(selected_component_image, frameNumber, file_dir)


def split_into_blocks(image, frameNumber, file_dir, block_size=(5, 5)):
    """Draws rectangle lines on the image to represent blocks."""
    block_height, block_width = block_size

    # Gambar blok sesuai dengan block_size jika memenuhi minimal 5x5
    for i in range(0, image.shape[0], block_height):
        for j in range(0, image.shape[1], block_width):
            if i + block_height <= image.shape[0] and j + block_width <= image.shape[1]:
                # add label to the block for j
                cv2.rectangle(
                    image,
                    (j, i),
                    (j + block_width, i + block_height),
                    1,
                )

    # Save the whole image with block lines
    cv2.imwrite(
        os.path.join(file_dir, f"annotated_block_frame-{frameNumber:02}.jpg"),
        image,
    )
