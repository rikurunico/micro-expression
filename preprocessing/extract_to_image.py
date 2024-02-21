from typing import Literal, Tuple
import os, cv2

def extract_component_as_image(image, frameNumber, objectRectangle: Tuple[int, int, int, int], objectName: Literal["mouth", "eye_left", "eye_right", "eyebrow_left", "eyebrow_right", "nose_right", "nose_left"]):
    file_dir = "dataset/element_images"
    # Buat path directory jika folder/file pathnya tidak ada
    os.makedirs(f"{file_dir}/{frameNumber}", exist_ok=True)
    # Ekstrak koordinat dari parameter coordinates
    y_top, x_right, y_bottom, x_left = objectRectangle
    # Seleksi gambar berdasarkan koordinat yang berhasil di ekstrak
    selected_component_image = image[y_top:y_bottom, x_left:x_right]
    #  Write gambar ke jpg sesuai dengan direktor nama component dan frame keberapa sekarang
    cv2.imwrite(f"{file_dir}/{frameNumber}/{frameNumber}-{objectName}.jpg", selected_component_image)