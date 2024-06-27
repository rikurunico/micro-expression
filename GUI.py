import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import threading
import joblib
import os
import pandas as pd
import dlib
import cv2
from collections import Counter
from helper.helper import format_number_and_round
from preprocessing.scarpping_component import extract_component_by_images
from feature_extraction.poc import POC
from feature_extraction.vektor import Vektor
from feature_extraction.quadran import Quadran
from preprocessing.input_to_image import get_frames_by_input_video
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image, ImageTk

# Memuat model dan label encoder
model_hidung = joblib.load("models/knn_model_dengan_hidung.joblib")
label_encoder_hidung = joblib.load("models/label_encoder_dengan_hidung.joblib")

model_tanpa_hidung = joblib.load("models/knn_model_tanpa_hidung.joblib")
label_encoder_tanpa_hidung = joblib.load("models/label_encoder.joblib")

components_setup = {
    "mulut": {
        "object_name": "mouth",
        "object_rectangle": {
            "x_right": 54,
            "x_left": 48,
            "y_highest": 52,
            "y_lowest": 57,
        },
        "pixel_shifting": {"pixel_x": 25, "pixel_y": 5},
        "object_dimension": {"width": 140, "height": 40},
    },
    "mata_kiri": {
        "object_name": "eye_left",
        "object_rectangle": {
            "x_right": 39,
            "x_left": 36,
            "y_highest": 38,
            "y_lowest": 41,
        },
        "pixel_shifting": {"pixel_x": 25, "pixel_y": 25},
        "object_dimension": {"width": 90, "height": 55},
    },
    "mata_kanan": {
        "object_name": "eye_right",
        "object_rectangle": {
            "x_right": 45,
            "x_left": 42,
            "y_highest": 43,
            "y_lowest": 47,
        },
        "pixel_shifting": {"pixel_x": 25, "pixel_y": 25},
        "object_dimension": {"width": 90, "height": 55},
    },
    "alis_kiri": {
        "object_name": "eyebrow_left",
        "object_rectangle": {
            "x_right": 21,
            "x_left": 17,
            "y_highest": 18,
            "y_lowest": 21,
        },
        "pixel_shifting": {"pixel_x": 15, "pixel_y": 15},
        "object_dimension": {"width": 110, "height": 40},
    },
    "alis_kanan": {
        "object_name": "eyebrow_right",
        "object_rectangle": {
            "x_right": 26,
            "x_left": 22,
            "y_highest": 25,
            "y_lowest": 22,
        },
        "pixel_shifting": {"pixel_x": 15, "pixel_y": 15},
        "object_dimension": {"width": 110, "height": 40},
    },
    "hidung_kanan": {
        "object_name": "nose_right",
        "object_rectangle": {
            "x_right": 31,
            "x_left": 40,
            "y_highest": 40,
            "y_lowest": 48,
        },
        "pixel_shifting": {"pixel_x": 15, "pixel_y": -25},
        "object_dimension": {"width": 70, "height": 40},
    },
    "hidung_kiri": {
        "object_name": "nose_left",
        "object_rectangle": {
            "x_right": 47,
            "x_left": 35,
            "y_highest": 47,
            "y_lowest": 54,
        },
        "pixel_shifting": {"pixel_x": 15, "pixel_y": -25},
        "object_dimension": {"width": 70, "height": 40},
    },
}

# Inisialisasi face detector dan shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")


# Fungsi untuk menghitung hasil prediksi
def get_calculate_from_predict(list_decoded_predictions):
    prediction_counts = Counter(list_decoded_predictions)
    total_predictions = len(list_decoded_predictions)
    result_prediction = None
    most_common_count = 0
    list_predictions = []

    for category, count in prediction_counts.items():
        percentage = (count / total_predictions) * 100
        list_predictions.append(
            {
                "name": category,
                "count": count,
                "percentage": format_number_and_round(percentage),
            }
        )
        if count > most_common_count:
            most_common_count = count
            result_prediction = category

    return result_prediction, list_predictions


# Fungsi untuk memproses video dan membuat prediksi
def process_video(video_path, output_image_directory, print_image=False):

    # Print nama file video
    print(f"Processing video: {video_path}")

    fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
    gambar = get_frames_by_input_video(
        video_path, output_image_directory, framePerSecond=fps
    )

    block_size = 5
    frames_data_all_component = []
    frames_data_quadran_column = ["sumX", "sumY", "Tetha", "Magnitude", "JumlahQuadran"]
    quadran_dimensions = ["Q1", "Q2", "Q3", "Q4"]
    frames_data = {component_name: [] for component_name in components_setup}
    total_blocks_components = {component_name: 0 for component_name in components_setup}
    data_blocks_first_image = {
        component_name: None for component_name in components_setup
    }
    index = {component_name: 0 for component_name in components_setup}

    for filename in os.listdir(output_image_directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = cv2.imread(os.path.join(output_image_directory, filename))
            image = cv2.resize(image, (600, 500))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray)

            if len(rects) == 0:
                continue

            for component_name in components_setup:
                if index[component_name] != 0:
                    frame_data_all_component = {
                        "Frame": f"{index[component_name] + 1}({filename.split('.')[0]})"
                    }
                    frame_data_quadran = {
                        "Frame": f"{index[component_name] + 1}({filename.split('.')[0]})"
                    }

            for rect in rects:
                shape = predictor(gray, rect)
                for component_name, component_info in components_setup.items():
                    sum_data_by_quadran = {}
                    frame_data = {
                        "Frame": f"{index[component_name] + 1}({filename.split('.')[0]})"
                    }
                    for column in frames_data_quadran_column:
                        sum_data_by_quadran[column] = {
                            quadrant: 0 for quadrant in quadran_dimensions
                        }

                    data_blocks_image_current = extract_component_by_images(
                        image=image,
                        shape=shape,
                        frameName=filename.split(".")[0],
                        objectName=component_info["object_name"],
                        objectRectangle=component_info["object_rectangle"],
                        pixelShifting=component_info["pixel_shifting"],
                        objectDimension=component_info["object_dimension"],
                    )

                    x_right = shape.part(
                        component_info["object_rectangle"]["x_right"]
                    ).x
                    x_left = shape.part(component_info["object_rectangle"]["x_left"]).x
                    y_highest = shape.part(
                        component_info["object_rectangle"]["y_highest"]
                    ).y
                    y_lowest = shape.part(
                        component_info["object_rectangle"]["y_lowest"]
                    ).y

                    color = (0, 255, 0)  # Green color for rectangle
                    thickness = 2

                    cv2.rectangle(
                        image,
                        (
                            x_left - component_info["pixel_shifting"]["pixel_x"],
                            y_highest - component_info["pixel_shifting"]["pixel_y"],
                        ),
                        (
                            x_right + component_info["pixel_shifting"]["pixel_x"],
                            y_lowest + component_info["pixel_shifting"]["pixel_y"],
                        ),
                        color,
                        thickness,
                    )

                    if data_blocks_first_image[component_name] is None:
                        frames_data[component_name].append(frame_data)
                        frame_data["Folder Path"] = "data_test"
                        frame_data["Label"] = "data_test"
                        data_blocks_first_image[component_name] = (
                            data_blocks_image_current
                        )
                        continue

                    init_poc = POC(
                        data_blocks_first_image[component_name],
                        data_blocks_image_current,
                        block_size,
                    )
                    val_poc = init_poc.getPOC()

                    init_quiv = Vektor(val_poc, block_size)
                    quiv_data = init_quiv.getVektor()

                    if print_image:
                        plt.imshow(np.uint8(data_blocks_image_current), cmap="gray")
                        plt.quiver(
                            quiv_data[:, 0],
                            quiv_data[:, 1],
                            quiv_data[:, 2],
                            quiv_data[:, 3],
                            scale=1,
                            scale_units="xy",
                            angles="xy",
                            color="g",
                        )

                        # Hapus Directory jika sudah ada sesuai directory yang diinginkan
                        dir_image = "output_image_5x5"
                        os.makedirs(f"{dir_image}/{component_name}", exist_ok=True)

                        try:
                            plt.savefig(
                                f"{dir_image}/{component_name}/{filename.split('.')[0]}.png"
                            )
                        except PermissionError as e:
                            print(f"Error: {e}")
                            continue
                        plt.clf()

                    init_quadran = Quadran(quiv_data)
                    quadran = init_quadran.getQuadran()

                    for i, quad in enumerate(quadran):
                        frame_data[f"X{i+1}"] = quad[1]
                        frame_data[f"Y{i+1}"] = quad[2]
                        frame_data[f"Tetha{i+1}"] = quad[3]
                        frame_data[f"Magnitude{i+1}"] = quad[4]

                        frame_data_all_component[f"{component_name}-X{i+1}"] = quad[1]
                        frame_data_all_component[f"{component_name}-Y{i+1}"] = quad[2]
                        frame_data_all_component[f"{component_name}-Tetha{i+1}"] = quad[
                            3
                        ]
                        frame_data_all_component[f"{component_name}-Magnitude{i+1}"] = (
                            quad[4]
                        )

                        if quad[5] in quadran_dimensions:
                            sum_data_by_quadran["sumX"][quad[5]] += quad[1]
                            sum_data_by_quadran["sumY"][quad[5]] += quad[2]
                            sum_data_by_quadran["Tetha"][quad[5]] += quad[3]
                            sum_data_by_quadran["Magnitude"][quad[5]] += quad[4]
                            sum_data_by_quadran["JumlahQuadran"][quad[5]] += 1

                    frames_data[component_name].append(frame_data)
                    frame_data["Folder Path"] = "data_test"
                    frame_data["Label"] = "data_test"

                    for quadran in quadran_dimensions:
                        for feature in frames_data_quadran_column:
                            column_name = f"{component_name}{feature}{quadran}"
                            frame_data_quadran[column_name] = sum_data_by_quadran[
                                feature
                            ][quadran]

            if index[component_name] != 0:
                frames_data_all_component.append(frame_data_all_component)
                frame_data_all_component["Folder Path"] = "data_test"
                frame_data_all_component["Label"] = "data_test"

            index[component_name] += 1

            cv2.imshow("Video", image)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            

    cv2.destroyAllWindows()
    return frames_data_all_component


# Fungsi untuk memprediksi dan mencetak hasil
def predict_and_print_results(
    df, model, label_encoder, feature_columns, with_nose=True
):
    df = df.drop(columns=feature_columns)
    predictions = model.predict(df.values)
    decoded_predictions = label_encoder.inverse_transform(predictions)
    result_prediction, list_predictions = get_calculate_from_predict(
        decoded_predictions
    )

    results = ""
    for pred in list_predictions:
        results += f"Kategori: {pred['name']}, Jumlah: {pred['count']}, Persentase: {pred['percentage']:.2f}%\n"

    results += f"\nHasil Prediksi: {result_prediction}"

    if with_nose:
        print("Predictions with Nose:")
    else:
        print("\nPredictions without Nose:")
    print(results)
    return results


# Membuat jendela utama
root = tk.Tk()
root.title("Analisis Emosi Video")
root.geometry("800x600")

# Menetapkan tema
style = ttk.Style()
style.theme_use("clam")


# Fungsi untuk memilih file video
def select_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.avi *.mp4")])
    if video_path:
        progress.start()
        start_time = time.time()
        btn_select_video.config(state=tk.DISABLED)
        check_print_image.config(state=tk.DISABLED)
        update_elapsed_time(start_time)
        threading.Thread(
            target=process_and_predict, args=(video_path, start_time)
        ).start()


def process_and_predict(video_path, start_time):
    try:
        output_image_directory = "output_image_5x5"
        frames_data_all_component = process_video(
            video_path, output_image_directory, print_image=var_print_image.get()
        )
        df_fitur_all = pd.DataFrame(frames_data_all_component)
        except_feature_columns = ["Frame", "Folder Path", "Label"]

        # Prediksi dan tampilkan hasil dengan hidung
        results_with_nose = predict_and_print_results(
            df_fitur_all.copy(),
            model_hidung,
            label_encoder_hidung,
            except_feature_columns,
        )
        text_results_with_nose.delete("1.0", tk.END)
        text_results_with_nose.insert(tk.END, results_with_nose)

        # Prediksi dan tampilkan hasil tanpa hidung
        nose_features = [col for col in df_fitur_all.columns if "hidung" in col]
        df_fitur_tanpa_hidung = df_fitur_all.drop(columns=nose_features)
        results_without_nose = predict_and_print_results(
            df_fitur_tanpa_hidung.copy(),
            model_tanpa_hidung,
            label_encoder_tanpa_hidung,
            except_feature_columns,
            with_nose=False,
        )
        text_results_without_nose.delete("1.0", tk.END)
        text_results_without_nose.insert(tk.END, results_without_nose)
    finally:
        progress.stop()
        btn_select_video.config(state=tk.NORMAL)
        check_print_image.config(state=tk.NORMAL)

        # Hitung dan tampilkan waktu yang telah berlalu
        elapsed_time = time.time() - start_time
        elapsed_time_label.config(text=f"Waktu Pemrosesan: {elapsed_time:.2f} detik")


def update_elapsed_time(start_time):
    elapsed_time = time.time() - start_time
    elapsed_time_label.config(text=f"Waktu Pemrosesan: {elapsed_time:.2f} detik")
    if (
        progress["value"] != 0
    ):  # Jika progress bar berjalan, perbarui waktu lagi setelah 100ms
        root.after(100, update_elapsed_time, start_time)


# Tambahkan frame untuk konten utama
frame_main = ttk.Frame(root, padding="20")
frame_main.pack(fill="both", expand=True)

# Tambahkan tombol untuk memilih file video
btn_select_video = ttk.Button(frame_main, text="Pilih Video", command=select_video)
btn_select_video.pack(pady=20)

# Tambahkan checkbox untuk memilih apakah akan menghasilkan dan menyimpan plot quiver
var_print_image = tk.BooleanVar()
check_print_image = ttk.Checkbutton(
    frame_main, text="Hasilkan dan Simpan Plot Quiver", variable=var_print_image
)
check_print_image.pack()

# Tambahkan notebook untuk hasil bertab
notebook = ttk.Notebook(frame_main)
notebook.pack(fill="both", expand=True)

# Tambahkan widget teks untuk menampilkan hasil di tab
frame_with_nose = ttk.Frame(notebook)
frame_without_nose = ttk.Frame(notebook)
notebook.add(frame_with_nose, text="Hasil dengan Hidung")
notebook.add(frame_without_nose, text="Hasil tanpa Hidung")

text_results_with_nose = tk.Text(frame_with_nose, height=15, width=80)
text_results_with_nose.pack(pady=10, padx=10)
text_results_with_nose.insert(tk.END, "Hasil dengan Hidung:\n")

text_results_without_nose = tk.Text(frame_without_nose, height=15, width=80)
text_results_without_nose.pack(pady=10, padx=10)
text_results_without_nose.insert(tk.END, "Hasil tanpa Hidung:\n")

# Tambahkan progress bar
progress = ttk.Progressbar(root, mode="indeterminate")
progress.pack(pady=20)

# Tambahkan label untuk menampilkan waktu yang telah berlalu
elapsed_time_label = ttk.Label(root, text="Waktu Pemrosesan: 0.00 detik")
elapsed_time_label.pack(pady=10)

# Mulai loop event utama
root.mainloop()
