import numpy as np
from typing import Literal, TypedDict
from preprocessing.extract_to_image import extract_component_as_image


class ObjectRectangle(TypedDict):
    x_right: int
    x_left: int
    y_highest: int
    y_lowest: int


class PixelShifting(TypedDict):
    pixel_x: int
    pixel_y: int


class ObjectDimension(TypedDict):
    width: int
    height: int


def extract_component_by_images(
    image,
    shape,
    frameName,
    objectName: Literal[
        "mouth",
        "eye_left",
        "eye_right",
        "eyebrow_left",
        "eyebrow_right",
        "nose_right",
        "nose_left",
    ],
    objectRectangle: ObjectRectangle,
    pixelShifting: PixelShifting,
    objectDimension: ObjectDimension,
):
    # Setup shape part dari parameter objectRectangle
    x_right = shape.part(objectRectangle["x_right"]).x
    x_left = shape.part(objectRectangle["x_left"]).x
    y_highest = shape.part(objectRectangle["y_highest"]).y
    y_lowest = shape.part(objectRectangle["y_lowest"]).y

    # Setup shape part dari parameter pixelShifting
    # Menggeser tepi kiri sisi gambar sebanyak variabel pergeseran_pixel ke kiri
    x_left -= pixelShifting["pixel_x"]
    # Menggeser tepi atas sisi gambar sebanyak variabel pergeseran_pixel ke atas
    y_highest -= pixelShifting["pixel_y"]

    # Memastikan koordinat tetap berada dalam batas size gambar
    x_left = max(0, x_left)
    y_highest = max(0, y_highest)
    width_object = min(objectDimension["width"], image.shape[1] - x_left)
    height_object = min(objectDimension["height"], image.shape[0] - y_highest)

    print(f"width_object: {width_object}, height_object: {height_object}")

    block_data = np.array(
        extract_component_as_image(
            image,
            frameName,
            (y_highest, x_left + width_object, y_highest + height_object, x_left),
            objectName,
        )
    )

    return block_data
