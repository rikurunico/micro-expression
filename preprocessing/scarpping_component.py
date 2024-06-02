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
    # for i in range(objectStart, objectEnd):
    #     x = shape.part(i).x
    #     y = shape.part(i).y

    # # Print face landmark with label
    # label = "{}".format(i)
    # cv2.circle(image, (x, y), 4, (255, 0, 0), -1)
    # cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    # Setup shape part dari parameter objectRectangle
    x_right = shape.part(objectRectangle["x_right"]).x
    x_left = shape.part(objectRectangle["x_left"]).x
    y_highest = shape.part(objectRectangle["y_highest"]).y
    y_lowest = shape.part(objectRectangle["y_lowest"]).y

    width_object = x_right - x_left
    height_object = y_lowest - y_highest

    # Setup shape part dari parameter pixelShifting
    # Menggeser tepi kiri sisi gambar sebanyak variabel pergeseran_pixel ke kiri
    x_left -= pixelShifting["pixel_x"]
    # Menggeser tepi atas sisi gambar sebanyak variabel pergeseran_pixel ke atas
    y_highest -= pixelShifting["pixel_y"]
    # Menambahkan sebanyak variabel pergeseran_pixel ke lebar (sisi kiri dan kanan)
    # width_object += pixelShifting["pixel_x"] * 2
    # Menambahkan sebanyak variabel pergeseran_pixel ke tinggi (sisi atas dan bawah)
    # height_object += pixelShifting["pixel_y"] * 2

    # Menggambar sebuah persegi panjang di sekitar ROI dengan koordinat yang sudah dihitung
    # cv2.rectangle(image, (x_left, y_highest), (x_left + width_object, y_highest + height_object), (0, 255, 0), 2)
    # Memanggil fungsi ekstraksi gambar dengan parameter yang sesuai

    # Periksa objectName dengan if-elif-else
    # if objectName == "mouth":
    #     width_object = 140
    #     height_object = 42
    # elif objectName == "eye_left" or objectName == "eye_right":
    #     width_object = 91
    #     height_object = 56
    # elif objectName == "eyebrow_left" or objectName == "eyebrow_right":
    #     width_object = 112
    #     height_object = 42
    # elif objectName == "nose_left" or objectName == "nose_right":
    #     width_object = 30
    #     height_object = 40
    # else:
    #     print("Object name not recognized")

    # Memastikan koordinat tetap berada dalam batas size gambar
    x_left = max(0, x_left)
    y_highest = max(0, y_highest)
    width_object = min(objectDimension["width"], image.shape[1] - x_left)
    height_object = min(objectDimension["height"], image.shape[0] - y_highest)

    print(f"width_object: {width_object}, height_object: {height_object}")

    # Print coordinates and image size
    print(f"x_left: {x_left}, x_right: {x_left + width_object}")
    print(f"y_top: {y_highest}, y_bottom: {y_highest + height_object}")
    print(f"Image size: {image.shape[1]} x {image.shape[0]}")


    block_data = np.array(extract_component_as_image(
        image,
        frameName,
        (y_highest, x_left + width_object, y_highest + height_object, x_left),
        objectName,
    ))

    return block_data