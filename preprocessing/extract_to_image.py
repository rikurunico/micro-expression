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
    # block_size=7,
):
    # Construct directory path for saving images
    file_dir = f"dataset/component_to_images/{objectName}/{frameNumber:02}"

    # Create directory if it doesn't exist
    os.makedirs(file_dir, exist_ok=True)

    y_top, x_right, y_bottom, x_left = objectRectangle

    # Kalo mata ukurannya 56 height nya ,98 untuk widthnya
    # 56 / 7 = 8 array kebawah
    # 98 / 7 = 14 array kesamping
    
    # buatkan agar dari y top dan x left 

    # modulo y_top, x_right, y_bottom, x_left with block_size and loop with -1 until value of modulo 0
    # while y_top % block_size != 0:
    #     y_top -= 1

    # while x_right % block_size != 0:
    #     x_right -= 1

    # while y_bottom % block_size != 0:
    #     y_bottom -= 1

    # while x_left % block_size != 0:
    #     x_left -= 1
    width_object = x_right - x_left
    height_object = y_bottom - y_top

    # while width_object % block_size != 0:
    #     width_object -= 1

    # while height_object % block_size != 0:
    #     height_object -= 1

    selected_component_image = image.copy()

    # Draw rectangle around the selected component
    # cv2.rectangle(
    #     selected_component_image,
    #     (x_left, y_top),
    #     (x_left + width_object, y_top + height_object),
    #     (0, 255, 0),
    #     1,
    # )
    # print("Print after resize", width_object, height_object)
    # Crop the selected component
    selected_component_image = selected_component_image[y_top:y_top + height_object + 1, x_left:x_left + width_object + 1]

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
    # blocks_data = split_into_blocks(selected_component_image_gray, file_dir, frameNumber, block_size)
    return selected_component_image_gray

# def split_into_blocks(image, file_dir, frameNumber, block_size):
#     """Returns a 2D array containing pixel values for each block in the image."""
#     blocks = []
#     for i in range(0, image.shape[0], block_size):
#         row = []
#         for j in range(0, image.shape[1], block_size):
#             if i + block_size <= image.shape[0] and j + block_size <= image.shape[1]:
#                 # Crop the block
#                 block = image[i:i + block_size, j:j + block_size]
#                 cv2.rectangle(
#                     image,
#                     (j, i),
#                     (j + block_size, i + block_size),
#                     1,
#                 )
#                 # Append the grayscale pixel values of the block
#                 row.append(round(np.mean(block)))
#         blocks.append(row)
#     cv2.imwrite(os.path.join(file_dir, f"annotated_frame_block-{frameNumber:02}.jpg"), image, )
#     return blocks

def split_into_blocks(image, file_dir, frameNumber, block_size):
    """Returns a 2D array containing pixel values for each block in the image."""
    blocks = []
    for i in range(0, image.shape[0], block_size):
        row = []
        for j in range(0, image.shape[1], block_size):
            if i + block_size <= image.shape[0] and j + block_size <= image.shape[1]:
                # Crop the block
                block = image[i:i + block_size, j:j + block_size]
                # Append the grayscale pixel values of the block as a numpy array
                row.append(np.array(block))
                # cv2.rectangle(
                #     image,
                #     (j, i),
                #     (j + block_size, i + block_size),
                #     1,
                # )
        blocks.append(row)
    # cv2.imwrite(os.path.join(file_dir, f"annotated_frame_block-{frameNumber:02}.jpg"), image)
    return blocks