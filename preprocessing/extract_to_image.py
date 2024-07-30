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
    file_dir = f"dataset/component_to_images/{objectName}/{frameNumber:02}"

    # Create directory if it doesn't exist
    os.makedirs(file_dir, exist_ok=True)

    y_top, x_right, y_bottom, x_left = objectRectangle

    width_object = x_right - x_left
    height_object = y_bottom - y_top

    selected_component_image = image.copy()

    # Crop the selected component
    selected_component_image = selected_component_image[
        y_top : y_top + height_object + 1, x_left : x_left + width_object + 1
    ]

    # Grayscale the image
    selected_component_image_gray = cv2.cvtColor(
        selected_component_image, cv2.COLOR_BGR2GRAY
    )

    # Save the annotated image
    cv2.imwrite(
        os.path.join(file_dir, f"annotated_frame-{frameNumber:02}.jpg"),
        selected_component_image_gray,
    )


    # Get pixel values for blocks
    return selected_component_image_gray
