import os
import time
import cv2
import numpy as np
from PySide6.QtCore import Qt, Slot, QSize
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QGroupBox, QPushButton, QComboBox, QLineEdit, QMessageBox,
    QListWidget, QListWidgetItem, QMenu, QInputDialog, QSlider, QCheckBox
)
from PySide6.QtGui import QPixmap, QImage

import pv_visionlib
import config_ini
from components.models import SimpleCNN

try:
    import tensorflow as tf
    from tensorflow import keras
    from ultralytics import YOLO # For our custom detector
    from components.resultsview import ResultImageItem # Re-use this handy widget
except ImportError:
    tf = None

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    import keras_ocr
except ImportError:
    keras_ocr = None


class InferenceResultItem(QWidget):
    """Custom widget for displaying a single character inference result."""
    def __init__(self, image_array, predicted_label, confidence, parent=None):
        super().__init__(parent)
        self.predicted_label = predicted_label
        self.image_array = image_array
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # Image Label
        self.image_label = QLabel()
        if image_array is not None:
            # Ensure the numpy array is C-contiguous before passing to QImage
            contiguous_array = np.ascontiguousarray(image_array)
            height, width = contiguous_array.shape[:2]
            if contiguous_array.ndim == 3:
                q_image = QImage(contiguous_array.data, width, height, 3 * width, QImage.Format.Format_RGB888).rgbSwapped()
            else:
                q_image = QImage(contiguous_array.data, width, height, width, QImage.Format.Format_Grayscale8)
            
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(48, 48, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
        else:
            self.image_label.setText("N/A")
            self.image_label.setFixedSize(48, 48)
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Info Layout
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(2)

        self.predicted_label_text = QLabel(f"Predicted: <b>{predicted_label}</b>")
        confidence_text = QLabel(f"Confidence: {confidence:.2f}%")

        if confidence < 70: # Highlight low-confidence predictions
            self.predicted_label_text.setStyleSheet("color: orange;")

        info_layout.addWidget(self.predicted_label_text)
        info_layout.addWidget(confidence_text)

        layout.addWidget(self.image_label)
        layout.addLayout(info_layout)
        layout.addStretch()
        self.setLayout(layout)

    def update_label(self, new_label):
        self.predicted_label = new_label
        self.predicted_label_text.setText(f"Predicted: <b>{new_label}</b>")
        self.predicted_label_text.setStyleSheet("color: green;")

class InferenceView(QWidget):
    """
    A widget for running inference on an image using a trained model.
    """
    def __init__(self, parent_wnd=None):
        super().__init__()
        self.parent_wnd = parent_wnd
        self.inference_model = None
        self.detector_model = None # Our new custom detector model
        self.class_names = None
        self.device = None

        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- Controls Group ---
        controls_group = QGroupBox("Controles de Inferência")
        controls_group.setContentsMargins(10, 20, 10, 10)
        controls_layout = QHBoxLayout(controls_group)
        controls_layout.setContentsMargins(10, 20, 10, 10)

        self.camera_select = QComboBox()
        self.camera_select.addItems(["Camera A", "Camera B"])

        self.library_select = QComboBox()
        self.library_select.addItems(["TensorFlow", "PyTorch", "EasyOCR", "KerasOCR"])
        self.library_select.currentTextChanged.connect(self.on_library_changed)

        self.run_inference_btn = QPushButton("Executar Inferência")
        self.run_inference_btn.clicked.connect(self.run_inference)
        self.run_inference_btn.setEnabled(False) # Disabled until a model is loaded

        self.save_btn = QPushButton("Salvar no Dataset")
        self.save_btn.clicked.connect(self.save_to_dataset)
        self.save_btn.setEnabled(False)

        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(0)
        self.confidence_label = QLabel("Confiança: 0%")
        self.confidence_slider.valueChanged.connect(lambda v: self.confidence_label.setText(f"Confiança: {v}%"))

        self.use_crop_checkbox = QCheckBox("Recortar área (Detector)")
        self.flip_image_checkbox = QCheckBox("Inverter Horizontalmente")

        controls_layout.addWidget(QLabel("Fonte da Imagem:"))
        controls_layout.addWidget(self.camera_select)
        controls_layout.addWidget(QLabel("Biblioteca:"))
        controls_layout.addWidget(self.library_select)
        controls_layout.addWidget(self.run_inference_btn)
        controls_layout.addWidget(self.save_btn)
        controls_layout.addWidget(self.use_crop_checkbox)
        controls_layout.addWidget(self.flip_image_checkbox)
        controls_layout.addWidget(self.confidence_label)
        controls_layout.addWidget(self.confidence_slider)
        controls_layout.addStretch()

        # --- Middle section for results and characters ---
        middle_layout = QHBoxLayout()
        middle_layout.setSpacing(10)

        # Left side: Combined results
        left_group = QGroupBox("Resultados da Inferência")
        left_group.setContentsMargins(10, 10, 10, 10)
        left_layout = QVBoxLayout(left_group)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)

        text_results_layout = QFormLayout()
        text_results_layout.setContentsMargins(10, 20, 10, 10)
        self.ground_truth_edit = QLineEdit()
        self.ground_truth_edit.setPlaceholderText("Insira os caracteres reais para comparação")
        self.predicted_text_label = QLabel("<b>N/A</b>")
        self.accuracy_label = QLabel("<b>N/A</b>")
        text_results_layout.addRow("Real:", self.ground_truth_edit)
        text_results_layout.addRow("Previsto:", self.predicted_text_label)
        text_results_layout.addRow("Acurácia:", self.accuracy_label)

        self.result_image_label = QLabel("Execute a inferência para ver a imagem resultante.")
        self.result_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_image_label.setMinimumSize(400, 150)
        self.result_image_label.setStyleSheet("border: 1px solid gray; background-color: #222;")

        left_layout.addLayout(text_results_layout)
        left_layout.addWidget(self.result_image_label)

        # Right side: Character by character predictions
        right_group = QGroupBox("Previsões por Caractere")
        right_layout = QVBoxLayout(right_group)
        right_layout.setContentsMargins(10, 20, 10, 10)
        right_layout.setSpacing(20)

        self.char_list_widget = QListWidget()
        self.char_list_widget.setFlow(QListWidget.Flow.LeftToRight)
        self.char_list_widget.setWrapping(True)
        self.char_list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.char_list_widget.setSpacing(5)
        self.char_list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.char_list_widget.customContextMenuRequested.connect(self.show_context_menu)
        right_layout.addWidget(self.char_list_widget)

        middle_layout.addWidget(left_group, 1)
        middle_layout.addWidget(right_group, 2)

        main_layout.addWidget(controls_group)
        main_layout.addLayout(middle_layout)
        main_layout.addStretch()

        # Set default library from config
        current_lib = getattr(config_ini, 'recognition_library', 'tensorflow')
        if current_lib.lower() == 'pytorch':
             self.library_select.setCurrentText("PyTorch")
        else:
             self.library_select.setCurrentText("TensorFlow")

    @Slot()
    def on_model_loaded(self):
        """
        Called when a model is loaded in the main application.
        This method loads the Keras model and label encoder.
        """
        if not self.parent_wnd or not self.parent_wnd.model_json.model:
            self.run_inference_btn.setEnabled(False)
            return

        model_data = self.parent_wnd.model_json.model
        detector_model_path = model_data.detector_model_path
        
        # Determine recognition model path based on library
        library = self.library_select.currentText().lower()
        recognition_model_path = model_data.encoder_filename
        
        if library == 'pytorch':
            base, _ = os.path.splitext(recognition_model_path)
            recognition_model_path = base + ".pth"

        # Check if paths exist (skip check for EasyOCR/KerasOCR as they don't use this path)
        if library not in ['easyocr', 'kerasocr'] and (not recognition_model_path or not os.path.exists(recognition_model_path) or not detector_model_path or not os.path.exists(detector_model_path)):
            self.run_inference_btn.setEnabled(False)
            self.inference_model = None
            self.detector_model = None
            self.class_names = None
            print(f"Model not found. Recognition: {recognition_model_path}, Detector: {detector_model_path}")
            return

        try:
            # Load both models
            self.detector_model = YOLO(detector_model_path)
            
            if library == 'tensorflow':
                try:
                    self.inference_model = keras.models.load_model(recognition_model_path)
                    self.device = None
                except Exception as e:
                    self.inference_model = None
                    QMessageBox.warning(self, "Erro de Carregamento", 
                        f"Falha ao carregar modelo TensorFlow.\n"
                        f"Verifique se o arquivo '{recognition_model_path}' é um modelo TensorFlow válido.\n"
                        f"Erro: {e}")
            elif library == 'pytorch':
                if torch and SimpleCNN:
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    self.inference_model = SimpleCNN(len(model_data.model_classes), int(model_data.image_height), int(model_data.image_width)).to(self.device)
                    try:
                        self.inference_model.load_state_dict(torch.load(recognition_model_path, map_location=self.device))
                        self.inference_model.eval()
                    except Exception as e:
                        self.inference_model = None
                        QMessageBox.warning(self, "Erro de Carregamento", 
                            f"Falha ao carregar modelo PyTorch.\n"
                            f"Verifique se o arquivo '{recognition_model_path}' é um modelo PyTorch válido.\n"
                            f"Erro: {e}")
                        print(f"PyTorch load error: {e}")
                else:
                    print("PyTorch not installed.")
                    self.inference_model = None

            self.class_names = model_data.model_classes
            
            if library in ['easyocr', 'kerasocr']:
                self.run_inference_btn.setEnabled(True)
            else:
                self.run_inference_btn.setEnabled(self.inference_model is not None)
                
            print("Inference and Detector models loaded successfully.")
        except Exception as e:
            self.run_inference_btn.setEnabled(False)
            QMessageBox.critical(self, "Erro ao Carregar Modelo", f"Não foi possível carregar o modelo de inferência:\n{e}")

    @Slot()
    def on_library_changed(self, text):
        if text in ["TensorFlow", "PyTorch"]:
            self.on_model_loaded()
            
        if text in ["EasyOCR", "KerasOCR"]:
            self.run_inference_btn.setEnabled(True)
        else:
            self.run_inference_btn.setEnabled(self.inference_model is not None)

    @Slot()
    def run_inference(self):
        library = self.library_select.currentText()
        if library == "EasyOCR":
            self.run_inference_easyocr()
        elif library == "KerasOCR":
            self.run_inference_kerasocr()
        else:
            self.run_custom_inference(library)

    def get_clean_camera_frame(self, camera):
        """Resets the camera view to the original image and clears ROIs."""
        if camera.image_path and os.path.exists(camera.image_path):
            # Reload image to clear any burned-in annotations
            camera.load_image_path(camera.image_path)
        
        # Clear any interactive ROIs
        if camera.rois:
            camera.delete_all_rois(camera.rois)
            
        return camera.actual_image.copy() if camera.actual_image is not None else None

    def run_custom_inference(self, library):
        """
        Executes the character detection and recognition process.
        """
        if not self.detector_model or not self.inference_model or not self.class_names:
            QMessageBox.warning(self, "Modelo não Carregado", "Nenhum modelo de inferência foi carregado.")
            return

        # Ensure we have access to the main program view and cameras
        program_view = self.parent_wnd
        if not program_view or not hasattr(program_view, 'cameras'):
            curr = self.parent()
            while curr:
                if hasattr(curr, 'cameras'):
                    program_view = curr
                    break
                curr = curr.parent()

        if not program_view or not hasattr(program_view, 'cameras'):
            QMessageBox.warning(self, "Erro", "Não foi possível acessar as câmeras.")
            return

        cam_index = self.camera_select.currentIndex()
        if len(program_view.cameras) <= cam_index:
            return

        camera = program_view.cameras[cam_index]
        source_image = self.get_clean_camera_frame(camera)

        if source_image is None:
            QMessageBox.warning(self, "Sem Imagem", "Nenhuma imagem carregada na câmera selecionada.")
            return

        if self.flip_image_checkbox.isChecked():
            source_image = cv2.flip(source_image, 1)
            # Update camera image to flipped version so user sees what is being processed
            camera.actual_image = source_image
            camera.label.setImage(source_image)

        # --- STAGE 1: DETECTION using our custom YOLO model ---
        # The YOLO model will return a list of bounding boxes for detected characters.
        detection_results = self.detector_model(source_image, verbose=False)
        boxes = detection_results[0].boxes.xyxy.cpu().numpy() # Get boxes in xyxy format
        
        if len(boxes) == 0:
            self.predicted_text_label.setText("<b>Nenhum caractere detectado.</b>")
            self.char_list_widget.clear()
            self.result_image_label.setPixmap(QPixmap.fromImage(pv_visionlib.pvVisionLib().convert_qt_image(source_image)))
            return
        
        # Sort boxes from left to right for correct string order
        character_boxes = sorted(boxes, key=lambda box: box[0])

        vision_lib = pv_visionlib.pvVisionLib()

        if not character_boxes:
            self.predicted_text_label.setText("<b>Nenhum caractere detectado.</b>")
            return

        self.char_list_widget.clear()
        predicted_string = ""
        
        # Determine crop area if requested
        img_to_display = source_image.copy()
        display_offset_x, display_offset_y = 0, 0

        if self.use_crop_checkbox.isChecked():
            # Find bounding box of all detected characters
            all_boxes = np.array(character_boxes)
            min_x = np.min(all_boxes[:, 0])
            min_y = np.min(all_boxes[:, 1])
            max_x = np.max(all_boxes[:, 2])
            max_y = np.max(all_boxes[:, 3])

            # Add padding
            padding = 20
            h, w, _ = source_image.shape
            min_x = max(0, int(min_x - padding))
            min_y = max(0, int(min_y - padding))
            max_x = min(w, int(max_x + padding))
            max_y = min(h, int(max_y + padding))

            img_to_display = source_image[min_y:max_y, min_x:max_x].copy()
            display_offset_x, display_offset_y = min_x, min_y

        img_with_boxes = img_to_display.copy()
        model_data = self.parent_wnd.model_json.model
        img_h, img_w = model_data.image_height, model_data.image_width
        rois_dict = {}

        for i, box in enumerate(character_boxes):
            x_min, y_min, x_max, y_max = map(int, box)
            
            # --- Preprocessing to match training data ---
            gray_img = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
            char_img = gray_img[y_min:y_max, x_min:x_max]
            
            h, w = char_img.shape
            if h > w:
                pad = (h - w) // 2
                char_img = cv2.copyMakeBorder(char_img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[0])
            elif w > h:
                pad = (w - h) // 2
                char_img = cv2.copyMakeBorder(char_img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0])

            resized_char = cv2.resize(char_img, (img_w, img_h), interpolation=cv2.INTER_AREA)
            normalized_char = resized_char / 255.0
            
            predicted_char = "?"
            confidence = 0.0
            valid_prediction = False

            if library == 'TensorFlow' and self.inference_model:
                input_data = np.expand_dims(np.expand_dims(normalized_char, axis=-1), axis=0)
                prediction = self.inference_model.predict(input_data, verbose=0)
                if prediction.any():
                    predicted_idx = np.argmax(prediction)
                    predicted_char = self.class_names[predicted_idx]
                    confidence = np.max(prediction) * 100
                    valid_prediction = True
            elif library == 'PyTorch':
                if self.inference_model:
                    # Prepare input: (H, W, 1) -> (1, 1, H, W)
                    input_tensor = torch.from_numpy(normalized_char).unsqueeze(0).unsqueeze(0).float().to(self.device)
                    with torch.no_grad():
                        outputs = self.inference_model(input_tensor)
                        probs = torch.softmax(outputs, dim=1)
                        conf, pred_idx = torch.max(probs, 1)
                        predicted_char = self.class_names[pred_idx.item()]
                        confidence = conf.item() * 100
                        valid_prediction = True

            if valid_prediction:
                if confidence < self.confidence_slider.value():
                    continue

                predicted_string += predicted_char

            # Draw bounding box and label on the result image
            # Adjust coordinates for the displayed image (which might be cropped)
            disp_x_min = x_min - display_offset_x
            disp_y_min = y_min - display_offset_y
            disp_x_max = x_max - display_offset_x
            disp_y_max = y_max - display_offset_y

            cv2.rectangle(img_with_boxes, (disp_x_min, disp_y_min), (disp_x_max, disp_y_max), (0, 255, 0), 2)
            cv2.putText(img_with_boxes, predicted_char, (disp_x_min, disp_y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Store ROI for interactive camera view
            rois_dict[str(i)] = {
                "char": f"{predicted_char} ({confidence:.1f}%)",
                "confidence": confidence,
                "box": {"x": disp_x_min, "y": disp_y_min, "w": disp_x_max - disp_x_min, "h": disp_y_max - disp_y_min}
            }

            # Add individual result to the list widget
            item = QListWidgetItem(self.char_list_widget)
            custom_widget = InferenceResultItem(
                resized_char, # Show the preprocessed image
                predicted_char,
                confidence
            )
            item.setSizeHint(QSize(220, 60))
            self.char_list_widget.addItem(item)
            self.char_list_widget.setItemWidget(item, custom_widget)

        self.save_btn.setEnabled(True)
        # Update UI
        self.predicted_text_label.setText(f"<b>{predicted_string}</b>")

        self.update_accuracy(predicted_string)

        # Display the result image
        qt_image = vision_lib.convert_qt_image(img_with_boxes)
        self.result_image_label.setPixmap(qt_image.scaled(
            self.result_image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))

        # Load the image with results into the camera view
        if program_view and hasattr(program_view, 'cameras') and len(program_view.cameras) > cam_index:
            program_view.cameras[cam_index].actual_image = img_with_boxes
            program_view.cameras[cam_index].draw_rois(img_with_boxes, [])
            program_view.cameras[cam_index].draw_rois_dict(rois_dict)
            program_view.cameras[cam_index].image_chars = predicted_string

    @Slot()
    def run_inference_easyocr(self):
        if easyocr is None:
            QMessageBox.warning(self, "EasyOCR Missing", "A biblioteca EasyOCR não está instalada.")
            return

        # Ensure we have access to the main program view and cameras
        program_view = self.parent_wnd
        if not program_view or not hasattr(program_view, 'cameras'):
            curr = self.parent()
            while curr:
                if hasattr(curr, 'cameras'):
                    program_view = curr
                    break
                curr = curr.parent()

        if not program_view or not hasattr(program_view, 'cameras'):
            QMessageBox.warning(self, "Erro", "Não foi possível acessar as câmeras.")
            return

        cam_index = self.camera_select.currentIndex()
        if len(program_view.cameras) <= cam_index:
            return

        camera = program_view.cameras[cam_index]
        source_image = self.get_clean_camera_frame(camera)

        if source_image is None:
            QMessageBox.warning(self, "Sem Imagem", "Nenhuma imagem carregada na câmera selecionada.")
            return

        if self.flip_image_checkbox.isChecked():
            source_image = cv2.flip(source_image, 1)
            # Update camera image to flipped version so user sees what is being processed
            camera.actual_image = source_image
            camera.label.setImage(source_image)

        img_to_process = source_image
        # offset_x, offset_y = 0, 0 # No longer needed if we display the crop

        if self.use_crop_checkbox.isChecked():
            if not self.detector_model:
                 QMessageBox.warning(self, "Modelo não Carregado", "O modelo detector YOLO é necessário para o recorte.")
                 return
            
            # Run detector to find the text area
            detection_results = self.detector_model(source_image, verbose=False)
            boxes = detection_results[0].boxes.xyxy.cpu().numpy()

            if len(boxes) > 0:
                # Find bounding box of all detected characters
                min_x = np.min(boxes[:, 0])
                min_y = np.min(boxes[:, 1])
                max_x = np.max(boxes[:, 2])
                max_y = np.max(boxes[:, 3])

                # Add padding
                padding = 20
                h, w, _ = source_image.shape
                min_x = max(0, int(min_x - padding))
                min_y = max(0, int(min_y - padding))
                max_x = min(w, int(max_x + padding))
                max_y = min(h, int(max_y + padding))

                img_to_process = source_image[min_y:max_y, min_x:max_x].copy()
                # offset_x, offset_y = min_x, min_y
            else:
                 QMessageBox.warning(self, "Nenhum Caractere", "O detector não encontrou caracteres para recortar.")
                 return

        try:
            reader = easyocr.Reader(['en'], gpu=True)
            results = reader.readtext(img_to_process)
        except Exception as e:
            QMessageBox.critical(self, "Erro EasyOCR", f"Falha na inferência EasyOCR:\n{e}")
            return

        self.char_list_widget.clear()
        predicted_string = ""
        img_with_boxes = img_to_process.copy() # Draw on the image we processed (cropped or full)
        rois_dict = {}
        vision_lib = pv_visionlib.pvVisionLib()

        for i, (bbox, text, prob) in enumerate(results):
            # Coordinates are already relative to img_to_process
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            x_min = max(0, int(min(x_coords)))
            x_max = min(img_with_boxes.shape[1], int(max(x_coords)))
            y_min = max(0, int(min(y_coords)))
            y_max = min(img_with_boxes.shape[0], int(max(y_coords)))

            cv2.rectangle(img_with_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(img_with_boxes, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            predicted_string += text + " "

            char_img = img_to_process[y_min:y_max, x_min:x_max]
            # Convert RGB to BGR for InferenceResultItem which expects BGR (due to rgbSwapped)
            char_img_bgr = cv2.cvtColor(char_img, cv2.COLOR_RGB2BGR)

            item = QListWidgetItem(self.char_list_widget)
            custom_widget = InferenceResultItem(char_img_bgr, text, prob * 100)
            item.setSizeHint(QSize(220, 60))
            self.char_list_widget.addItem(item)
            self.char_list_widget.setItemWidget(item, custom_widget)

            rois_dict[str(i)] = {
                "char": f"{text} ({prob*100:.1f}%)",
                "confidence": prob * 100,
                "box": {"x": x_min, "y": y_min, "w": x_max - x_min, "h": y_max - y_min}
            }

        self.save_btn.setEnabled(True)
        self.predicted_text_label.setText(f"<b>{predicted_string.strip()}</b>")
        self.update_accuracy(predicted_string.strip())

        qt_image = vision_lib.convert_qt_image(img_with_boxes)
        self.result_image_label.setPixmap(qt_image.scaled(
            self.result_image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))

        program_view.cameras[cam_index].actual_image = img_with_boxes
        program_view.cameras[cam_index].draw_rois(img_with_boxes, [])
        program_view.cameras[cam_index].draw_rois_dict(rois_dict)
        program_view.cameras[cam_index].image_chars = predicted_string.strip()

    @Slot()
    def run_inference_kerasocr(self):
        if keras_ocr is None:
            QMessageBox.warning(self, "KerasOCR Missing", "A biblioteca keras-ocr não está instalada.")
            return

        # Ensure we have access to the main program view and cameras
        program_view = self.parent_wnd
        if not program_view or not hasattr(program_view, 'cameras'):
            curr = self.parent()
            while curr:
                if hasattr(curr, 'cameras'):
                    program_view = curr
                    break
                curr = curr.parent()

        if not program_view or not hasattr(program_view, 'cameras'):
            QMessageBox.warning(self, "Erro", "Não foi possível acessar as câmeras.")
            return

        cam_index = self.camera_select.currentIndex()
        if len(program_view.cameras) <= cam_index:
            return

        camera = program_view.cameras[cam_index]
        source_image = self.get_clean_camera_frame(camera)

        if source_image is None:
            QMessageBox.warning(self, "Sem Imagem", "Nenhuma imagem carregada na câmera selecionada.")
            return

        if self.flip_image_checkbox.isChecked():
            source_image = cv2.flip(source_image, 1)
            # Update camera image to flipped version so user sees what is being processed
            camera.actual_image = source_image
            camera.label.setImage(source_image)

        img_to_process = source_image
        
        if self.use_crop_checkbox.isChecked():
            if not self.detector_model:
                 QMessageBox.warning(self, "Modelo não Carregado", "O modelo detector YOLO é necessário para o recorte.")
                 return
            
            # Run detector to find the text area
            detection_results = self.detector_model(source_image, verbose=False)
            boxes = detection_results[0].boxes.xyxy.cpu().numpy()

            if len(boxes) > 0:
                # Find bounding box of all detected characters
                min_x = np.min(boxes[:, 0])
                min_y = np.min(boxes[:, 1])
                max_x = np.max(boxes[:, 2])
                max_y = np.max(boxes[:, 3])

                # Add padding
                padding = 20
                h, w, _ = source_image.shape
                min_x = max(0, int(min_x - padding))
                min_y = max(0, int(min_y - padding))
                max_x = min(w, int(max_x + padding))
                max_y = min(h, int(max_y + padding))

                img_to_process = source_image[min_y:max_y, min_x:max_x].copy()
            else:
                 QMessageBox.warning(self, "Nenhum Caractere", "O detector não encontrou caracteres para recortar.")
                 return

        try:
            pipeline = keras_ocr.pipeline.Pipeline()
            # Keras-ocr expects RGB
            img_rgb = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2RGB)
            prediction_groups = pipeline.recognize([img_rgb])
            predictions = prediction_groups[0]
        except Exception as e:
            QMessageBox.critical(self, "Erro KerasOCR", f"Falha na inferência KerasOCR:\n{e}")
            return

        self.char_list_widget.clear()
        predicted_string = ""
        img_with_boxes = img_to_process.copy()
        rois_dict = {}
        vision_lib = pv_visionlib.pvVisionLib()

        for i, (text, box) in enumerate(predictions):
            # box is [[x,y], [x,y], [x,y], [x,y]]
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x_min = max(0, int(min(xs)))
            x_max = min(img_with_boxes.shape[1], int(max(xs)))
            y_min = max(0, int(min(ys)))
            y_max = min(img_with_boxes.shape[0], int(max(ys)))

            cv2.rectangle(img_with_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(img_with_boxes, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            predicted_string += text + " "

            char_img = img_to_process[y_min:y_max, x_min:x_max]
            # Convert RGB to BGR for InferenceResultItem which expects BGR (due to rgbSwapped)
            char_img_bgr = cv2.cvtColor(char_img, cv2.COLOR_RGB2BGR)

            item = QListWidgetItem(self.char_list_widget)
            custom_widget = InferenceResultItem(char_img_bgr, text, 0.0)
            item.setSizeHint(QSize(220, 60))
            self.char_list_widget.addItem(item)
            self.char_list_widget.setItemWidget(item, custom_widget)

            rois_dict[str(i)] = {
                "char": f"{text}",
                "confidence": 0.0,
                "box": {"x": x_min, "y": y_min, "w": x_max - x_min, "h": y_max - y_min}
            }

        self.save_btn.setEnabled(True)
        self.predicted_text_label.setText(f"<b>{predicted_string.strip()}</b>")
        self.update_accuracy(predicted_string.strip())

        qt_image = vision_lib.convert_qt_image(img_with_boxes)
        self.result_image_label.setPixmap(qt_image.scaled(
            self.result_image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))

        program_view.cameras[cam_index].actual_image = img_with_boxes
        program_view.cameras[cam_index].draw_rois(img_with_boxes, [])
        program_view.cameras[cam_index].draw_rois_dict(rois_dict)
        program_view.cameras[cam_index].image_chars = predicted_string.strip()

    def show_context_menu(self, pos):
        item = self.char_list_widget.itemAt(pos)
        if item:
            menu = QMenu(self)
            delete_action = menu.addAction("Delete")
            correct_action = menu.addAction("Correct")
            
            action = menu.exec(self.char_list_widget.mapToGlobal(pos))
            
            if action == delete_action:
                self.delete_prediction(item)
            elif action == correct_action:
                self.correct_prediction(item)

    def delete_prediction(self, item):
        row = self.char_list_widget.row(item)
        self.char_list_widget.takeItem(row)
        self.update_full_prediction_label()

    def correct_prediction(self, item):
        widget = self.char_list_widget.itemWidget(item)
        if not widget: return
        
        text, ok = QInputDialog.getText(self, "Correct Prediction", "Enter correct character:", text=widget.predicted_label)
        if ok and text:
            widget.update_label(text)
            self.update_full_prediction_label()

    def update_accuracy(self, predicted_string):
        """Calculates and updates the accuracy label based on ground truth."""
        ground_truth = self.ground_truth_edit.text()
        if ground_truth:
            correct_chars = sum(1 for i in range(min(len(ground_truth), len(predicted_string))) if ground_truth[i] == predicted_string[i])
            accuracy = (correct_chars / len(ground_truth)) * 100 if len(ground_truth) > 0 else 0
            self.accuracy_label.setText(f"<b>{accuracy:.2f}%</b>")
        else:
            self.accuracy_label.setText("<b>N/A</b>")

    def update_full_prediction_label(self):
        """Reconstructs the full predicted string from the list items."""
        full_text = ""
        for i in range(self.char_list_widget.count()):
            item = self.char_list_widget.item(i)
            widget = self.char_list_widget.itemWidget(item)
            if widget:
                full_text += widget.predicted_label
        self.predicted_text_label.setText(f"<b>{full_text}</b>")
        self.update_accuracy(full_text)

    def save_to_dataset(self):
        """Saves the current inference results (images and labels) to the training dataset."""
        if not self.parent_wnd or not self.parent_wnd.model_json.model:
            return

        dataset_path = self.parent_wnd.model_json.model.model_train_dataset
        if not dataset_path or not os.path.exists(dataset_path):
            QMessageBox.warning(self, "Erro", "Caminho do dataset de treino não configurado ou inválido.")
            return

        count = 0
        for i in range(self.char_list_widget.count()):
            item = self.char_list_widget.item(i)
            widget = self.char_list_widget.itemWidget(item)
            if widget and widget.image_array is not None:
                label = widget.predicted_label
                # Create class folder if it doesn't exist
                class_dir = os.path.join(dataset_path, label)
                os.makedirs(class_dir, exist_ok=True)
                
                # Generate filename
                timestamp = int(time.time() * 1000)
                filename = f"{label}_{timestamp}_{i}.png"
                save_path = os.path.join(class_dir, filename)
                
                try:
                    cv2.imwrite(save_path, widget.image_array)
                    count += 1
                except Exception as e:
                    print(f"Failed to save {save_path}: {e}")

        QMessageBox.information(self, "Sucesso", f"{count} imagens salvas no dataset de treino.")