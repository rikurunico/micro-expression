from typing import Literal, TypedDict
from preprocessing.extract_to_image import extract_component_as_image

class ObjectRectangle(TypedDict):
    x_right: int
    x_left: int
    y_highest: int
    y_lowest: int

def extract_component_by_images(image, shape, frameName, objectName: Literal["mouth", "eye_left", "eye_right", "eyebrow_left", "eyebrow_right"], objectStart, objectEnd,  objectRectangle: ObjectRectangle, pergeseranPixel=0):
    print(f"\n{frameName}-{objectName.capitalize()}")

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

    # Menggeser tepi kiri sisi gambar sebanyak variabel pergeseran_pixel ke kiri
    x_left -= pergeseranPixel 
    # Menggeser tepi atas sisi gambar sebanyak variabel pergeseran_pixel ke atas
    y_highest -= pergeseranPixel  
    # Menambahkan sebanyak variabel pergeseran_pixel ke lebar (sisi kiri dan kanan)
    width_object += (pergeseranPixel * 2)  
    # Menambahkan sebanyak variabel pergeseran_pixel ke tinggi (sisi atas dan bawah)
    height_object += (pergeseranPixel * 2) 

    # Memastikan koordinat tetap berada dalam batas size gambar
    x_left = max(0, x_left)  
    y_highest = max(0, y_highest)  
    width_object = min(width_object, image.shape[1] - x_left)  
    height_object = min(height_object, image.shape[0] - y_highest) 

    # Menggambar sebuah persegi panjang di sekitar ROI dengan koordinat yang sudah dihitung
    # cv2.rectangle(image, (x_left, y_highest), (x_left + width_object, y_highest + height_object), (0, 255, 0), 2)
    # Memanggil fungsi ekstraksi gambar dengan parameter yang sesuai
    extract_component_as_image(image, frameName, (y_highest, x_left + width_object, y_highest + height_object, x_left), objectName)
    print("Width: {}, Height: {}".format(width_object, height_object))