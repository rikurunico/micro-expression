def store():
    request_data = DataTestStoreRequest()
    
    if not request_data.validate():
        return response.error(422, 'Invalid request form validation', request_data.errors)
    
    try:
        # Mendapatkan file dari request
        file = request_data.file.data
        filename = secure_filename(file.filename)
        
        # Mendapatkan ekstensi dari filename dengan split
        file_extension = filename.split('.')[-1].lower()
        
        # Misalnya, nama file baru tanpa ekstensi
        new_filename = f'video-{str(uuid.uuid4())}'
        
        # Menggabungkan new_filename dengan ekstensi dan buat path untuk output 
        new_filename_with_extension = f"{new_filename}.{file_extension}"
        file_path_video = os.path.join(app.config['UPLOAD_FOLDER'], app.config['UPLOAD_FOLDER_VIDEO'], new_filename_with_extension)
        file_path_output_images = os.path.join(app.config['UPLOAD_FOLDER'], app.config['UPLOAD_FOLDER_IMAGE'], 'output', new_filename)

        # Save video ke folder di lokal
        file.save(file_path_video)

        # Lakukan pengecekan berdasarkan file extension
        if file_extension == 'avi':
            # Convert AVI ke WEBM untuk respons
            converted_webm_filename = f"{new_filename}.webm"
            converted_webm_file_path = os.path.join(app.config['UPLOAD_FOLDER'], app.config['UPLOAD_FOLDER_VIDEO'], converted_webm_filename)
            file_path_video_response = convert_video_to_webm(file_path_video, converted_webm_file_path)
        elif file_extension == 'webm':
            # Convert WEBM ke AVI untuk pemrosesan
            converted_avi_filename = f"{new_filename}.avi"
            converted_avi_file_path = os.path.join(app.config['UPLOAD_FOLDER'], app.config['UPLOAD_FOLDER_VIDEO'], converted_avi_filename)
            convert_video_to_avi(file_path_video, converted_avi_file_path)
            file_path_video = converted_avi_file_path
            new_filename_with_extension = f"{new_filename}.avi"
            file_path_video_response = file_path_video
        else:
            # Convert input video ke AVI untuk pemrosesan
            converted_avi_filename = f"{new_filename}.avi"
            converted_avi_file_path = os.path.join(app.config['UPLOAD_FOLDER'], app.config['UPLOAD_FOLDER_VIDEO'], converted_avi_filename)
            convert_video_to_avi(file_path_video, converted_avi_file_path)
            file_path_video = converted_avi_file_path
            new_filename_with_extension = f"{new_filename}.avi"
            
            # Convert input video ke WEBM untuk respons
            converted_webm_filename = f"{new_filename}.webm"
            converted_webm_file_path = os.path.join(app.config['UPLOAD_FOLDER'], app.config['UPLOAD_FOLDER_VIDEO'], converted_webm_filename)
            file_path_video_response = convert_video_to_webm(file_path_video, converted_webm_file_path)
            
            # Hapus file input yang bukan AVI atau WEBM setelah konversi
            os.remove(file_path_video)

        images, error = get_frames_by_input_video(file_path_video, file_path_output_images, 200)
        if error is not None:
            return response.error(message=error)
        
        # Variabel untuk format response sucess output
        output_data = []

        # --- Setup untuk perhitungan POC dari output images ---
        # load model dan shape predictor untuk deteksi wajah
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(os.path.join(app.config['UPLOAD_FOLDER'], app.config['UPLOAD_FOLDER_MODEL'], MODEL_PREDICTOR))
        
        # Inisialisasi variabel untuk menyimpan data dari masing-masing komponen
        components_setup = COMPONENTS_SETUP
        quadran_dimensions = QUADRAN_DIMENSIONS
        frames_data_quadran_column = FRAMES_DATA_QUADRAN_COMPONENTS
        frames_data_quadran = []
        frames_data_all_component = []
        frames_data = {component_name: [] for component_name in components_setup}
        total_blocks_components = {component_name: 0 for component_name in components_setup}
        data_blocks_first_image = {component_name: None for component_name in components_setup}
        index = {component_name: 0 for component_name in components_setup}

        for component_name, component_info in components_setup.items():
            total_blocks_components[component_name] = int((component_info['object_dimension']['width'] / BLOCKSIZE) * (component_info['object_dimension']['height'] / BLOCKSIZE))

        for filename in sorted(os.listdir(file_path_output_images), key=natural_sort_key):
            if filename.endswith(".jpg") or filename.endswith(".png"): 
                image = cv2.imread(os.path.join(file_path_output_images, filename))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                rects = detector(gray)
                
                current_image_data = {
                    "name": filename,
                    "url": next((img['url'] for img in images if os.path.splitext(img['name'])[0] == os.path.splitext(filename)[0]), None),
                    "components": {}
                }

                if not index[component_name] == 0:
                    frame_data_all_component = {'Frame': f"{index[component_name] + 1}({filename.split('.')[0]})"}
                    frame_data_quadran = {'Frame': f"{index[component_name] + 1}({filename.split('.')[0]})"}

                for rect in rects:
                    shape = predictor(gray, rect)
                    for component_name, component_info in components_setup.items():
                        sum_data_by_quadran = {}
                        frame_data = {'Frame': f"{index[component_name] + 1}({filename.split('.')[0]})"}

                        for column in frames_data_quadran_column:
                            sum_data_by_quadran[column] = {quadrant: 0 for quadrant in quadran_dimensions}
                        
                        data_blocks_image_current, image_url = extract_component_by_images(
                            image=image,
                            shape=shape,
                            frameName=filename.split(".")[0], 
                            objectName=component_info['object_name'],
                            objectRectangle=component_info['object_rectangle'],
                            pixelShifting=component_info['pixel_shifting'],
                            objectDimension=component_info['object_dimension'],
                            directoryOutputImage=file_path_output_images
                        )

                        current_image_data["components"][component_name] = {
                            "url_source": image_url
                        }
                        
                        if data_blocks_first_image[component_name] is None:
                            frames_data[component_name].append(frame_data)
                            frame_data['Folder Path'] = "data_test"
                            frame_data['Label'] = "data_test"
                            data_blocks_first_image[component_name] = data_blocks_image_current
                            continue

                        # Inisiasi class POC
                        initPOC = POC(data_blocks_first_image[component_name], data_blocks_image_current, BLOCKSIZE) 
                        # Pemanggilan fungsi pocCalc() untuk menghitung nilai POC disetiap gambar
                        valPOC = initPOC.getPOC() 

                        # Pemanggilan class dan method untuk menampilkan quiver / gambar panah
                        initQuiv = Vektor(valPOC, BLOCKSIZE)
                        quivData = initQuiv.getVektor() 

                        # Pemanggilan class untuk mengeluarkan nilai karakteristik vektor dan quadran
                        initQuadran = Quadran(quivData) 
                        
                        quadran = initQuadran.getQuadran()

                        # Tampilkan gambar grayscale dengan quiver dan simpan plot nya
                        url_result = draw_quiver_and_save_plotlib_image(
                            dataBlockImage=data_blocks_image_current, 
                            quivData=quivData,
                            frameName=filename.split(".")[0],
                            objectName=component_info['object_name'], 
                            directoryOutputImage=file_path_output_images
                        )
                        
                        current_image_data["components"][component_name]["url_result"] = url_result

                        # print(tabulate(quadran, headers=['Blok Ke', 'X', 'Y', 'Tetha', 'Magnitude', 'Quadran Ke']))

                        # Update frame_data dengan data quadran
                        for i, quad in enumerate(quadran):
                            # --- Setup bagian Nilai Fitur Dataset ---
                            # Set data kedalam frame_data sesuai column nya
                            frame_data[f'X{i+1}'] = quad[1]
                            frame_data[f'Y{i+1}'] = quad[2]
                            frame_data[f'Tetha{i+1}'] = quad[3]
                            frame_data[f'Magnitude{i+1}'] = quad[4]

                            # Set data kedalam frame_data_all_component sesuai columnnya
                            frame_data_all_component[f'{component_name}-X{i+1}'] = quad[1]
                            frame_data_all_component[f'{component_name}-Y{i+1}'] = quad[2]
                            frame_data_all_component[f'{component_name}-Tetha{i+1}'] = quad[3]
                            frame_data_all_component[f'{component_name}-Magnitude{i+1}'] = quad[4]

                            # --- Setup bagian 4qmv Dataset ---
                            # Cek apakah quad[5] ada didalam array quadran_dimensions
                            if quad[5] in quadran_dimensions:
                                # Tambahkan nilai quad[1] ke sumX pada kuadran yang sesuai
                                sum_data_by_quadran['sumX'][quad[5]] += quad[1]
                                # Tambahkan nilai quad[2] ke sumY pada kuadran yang sesuai
                                sum_data_by_quadran['sumY'][quad[5]] += quad[2]
                                # Tambahkan nilai quad[3] ke Tetha pada kuadran yang sesuai
                                sum_data_by_quadran['Tetha'][quad[5]] += quad[3]
                                # Tambahkan nilai quad[4] ke Magnitude pada kuadran yang sesuai
                                sum_data_by_quadran['Magnitude'][quad[5]] += quad[4]
                                # Tambahkan jumlah quadran sesuai dengan quad[5] ke JumlahQuadran pada kuadran yang sesuai
                                sum_data_by_quadran['JumlahQuadran'][quad[5]] += 1
                        
                        # --- Setup bagian Nilai Fitur Dataset ---
                        # Append data frame ke list
                        frames_data[component_name].append(frame_data)
                        # Tambahkan kolom "Folder Path" dengan nilai folder saat ini
                        frame_data['Folder Path'] = "data_test"
                        # Tambahkan kolom "Label" dengan nilai label saat ini
                        frame_data['Label'] = "data_test"

                        # --- Setup bagian 4qmv Dataset ---
                        # Inisialisasi data untuk setiap blok dan setiap kuadran dengan nilai sesuai sum_data_by_quadran
                        for quadran in quadran_dimensions:
                            for feature in frames_data_quadran_column:
                                # Buat nama kolom dengan menggunakan template yang diberikan
                                column_name = f"{component_name}{feature}{quadran}"
                                # Set value sum_data_by_quadran[feature][quadran] ke frame_data_quadran sesuai column_name nya
                                frame_data_quadran[column_name] = sum_data_by_quadran[feature][quadran]

                if not index[component_name] == 0:
                    # --- Setup bagian 4qmv Dataset ---
                    # Append data frame ke list frames_data_quadran untuk 4qmv
                    frames_data_quadran.append(frame_data_quadran)
                    # Tambahkan kolom "Folder Path" dengan nilai folder saat ini
                    frame_data_quadran['Folder Path'] = "data_test"
                    # Tambahkan kolom "Label" dengan nilai label saat ini
                    frame_data_quadran['Label'] = "data_test"

                    # --- Setup bagian frames data all component Dataset ---
                    # Append data frame ke list frames_data_quadran untuk 4qmv
                    frames_data_all_component.append(frame_data_all_component)
                    # Tambahkan kolom "Folder Path" dengan nilai folder saat ini
                    frame_data_all_component['Folder Path'] = "data_test"
                    # Tambahkan kolom "Label" dengan nilai label saat ini
                    frame_data_all_component['Label'] = "data_test"

                # Update index per component_name
                index[component_name] += 1

                # Append current_image_data ke output_data
                output_data.append(current_image_data)

        svm_model_path = os.path.join(app.config['UPLOAD_FOLDER'], app.config['UPLOAD_FOLDER_MODEL'], 'svm_model.joblib')
        label_encoder_path = os.path.join(app.config['UPLOAD_FOLDER'], app.config['UPLOAD_FOLDER_MODEL'], 'label_encoder.joblib')
        svm_model = joblib.load(svm_model_path)
        label_encoder = joblib.load(label_encoder_path)
        
        df_fitur_all = pd.DataFrame(frames_data_all_component)
        except_feature_columns = ['Frame', 'Folder Path', 'Label']  
        
        df_fitur_all = df_fitur_all.drop(columns=except_feature_columns)

        # Lakukan prediksi dengan model yang telah dimuat
        predictions = svm_model.predict(df_fitur_all.values)

        # Ubah prediksi numerik menjadi label asli
        decoded_predictions = label_encoder.inverse_transform(predictions)
        
        result_prediction, list_predictions = get_calculate_from_predict(decoded_predictions)
        print("decoded_predictions : ", len(decoded_predictions))
        print("output_data : ", len(output_data))

        for i in range(len(output_data)):
            if i == 0:
                output_data[i]['prediction'] = None
            else:
                output_data[i]['prediction'] = decoded_predictions[i-1]

        # Return response sukses untuk date video dan images, dan prediction
        return response.success(200, 'Ok', {
            "video": {
                "url": file_path_video_response, 
                "name" : converted_webm_filename if file_extension != 'webm' else new_filename_with_extension
            },
            "result" : result_prediction,
            "list_predictions" : list_predictions,
            "images": output_data,
        })
    except Exception as e:
        return response.error(message=str(e))