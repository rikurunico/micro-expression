import cv2, os

# Buat fungsi agar bisa dipanggil di codebase main (nantinya)
def get_frames_by_input_video(
    pathInputVideo, pathOutputImage="dataset/video_to_images", framePerSecond=60
):
    # Cek apakah filenya terdeteksi
    if not os.path.exists(pathInputVideo):
        print(f"Path file {pathInputVideo} tidak valid")
        return

    # Buat path directory jika folder/file pathnya tidak ada
    os.makedirs(pathOutputImage, exist_ok=True)
    # Looping untuk menghapus file convert image didalam folder
    for filename in os.listdir(pathOutputImage):
        filepath = os.path.join(pathOutputImage, filename)
        os.remove(filepath)

    # Convert path video ke video capture
    vidcap = cv2.VideoCapture(pathInputVideo)

    # Buat variabel untuk looping while
    count = 1

    while True:
        # Read setiap frame dari video (setiap looping framenya bertambah 1 sampai jumlah frame video habis baru success bernilai false)
        success, image = vidcap.read()
        if not success:
            break

        #  Write gambar ke jpg
        cv2.imwrite(f"{pathOutputImage}/frame{count}.jpg", image)
        # count ++ untuk melanjutkan looping ke frame berikutnya
        count += 1
        # hitung waktu diambilnya frame ke sekian detik dari durasi video
        expected_frame_time = count / framePerSecond
        # set waktu frame yang diambil
        vidcap.set(cv2.CAP_PROP_POS_MSEC, expected_frame_time * 1000)