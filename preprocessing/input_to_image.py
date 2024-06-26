import cv2, os

# Buat fungsi agar bisa dipanggil di codebase main (nantinya)
def get_frames_by_input_video(
    pathInputVideo, pathOutputImage="dataset/video_to_images", framePerSecond=60
):
    if not os.path.exists(pathInputVideo):
        print(f"Path file {pathInputVideo} tidak valid")
        return
    os.makedirs(pathOutputImage, exist_ok=True)
    for filename in os.listdir(pathOutputImage):
        filepath = os.path.join(pathOutputImage, filename)
        os.remove(filepath)
    vidcap = cv2.VideoCapture(pathInputVideo)
    count = 1
    while True:
        success, image = vidcap.read()
        if not success:
            break
        cv2.imwrite(f"{pathOutputImage}/frame{count}.jpg", image)
        count += 1
        expected_frame_time = count / framePerSecond
        vidcap.set(cv2.CAP_PROP_POS_MSEC, expected_frame_time * 1000)