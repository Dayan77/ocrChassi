import os
import cv2
import numpy as np
from PySide6.QtCore import Qt, Slot, QSize
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QGroupBox, QPushButton, QComboBox, QLineEdit, QMessageBox,
    QListWidget, QListWidgetItem
)
from PySide6.QtGui import QPixmap, QImage

import pv_visionlib

try:
    import tensorflow as tf
    from tensorflow import keras
    from ultralytics import YOLO # For our custom detector
    from components.resultsview import ResultImageItem # Re-use this handy widget
except ImportError:
    tf = None

class InferenceResultItem(QWidget):
    """Custom widget for displaying a single character inference result."""
    def __init__(self, image_array, predicted_label, confidence, parent=None):
        super().__init__(parent)
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

        predicted_label_text = QLabel(f"Predicted: <b>{predicted_label}</b>")
        confidence_text = QLabel(f"Confidence: {confidence:.2f}%")

        if confidence < 70: # Highlight low-confidence predictions
            predicted_label_text.setStyleSheet("color: orange;")

        info_layout.addWidget(predicted_label_text)
        info_layout.addWidget(confidence_text)

        layout.addWidget(self.image_label)
        layout.addLayout(info_layout)
        layout.addStretch()
        self.setLayout(layout)

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

        self.run_inference_btn = QPushButton("Executar Inferência")
        self.run_inference_btn.clicked.connect(self.run_inference)
        self.run_inference_btn.setEnabled(False) # Disabled until a model is loaded

        controls_layout.addWidget(QLabel("Fonte da Imagem:"))
        controls_layout.addWidget(self.camera_select)
        controls_layout.addWidget(self.run_inference_btn)
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
        right_layout.addWidget(self.char_list_widget)

        middle_layout.addWidget(left_group, 1)
        middle_layout.addWidget(right_group, 2)

        main_layout.addWidget(controls_group)
        main_layout.addLayout(middle_layout)
        main_layout.addStretch()

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
        recognition_model_path = model_data.get_model_save_path()
        detector_model_path = model_data.detector_model_path

        if not recognition_model_path or not os.path.exists(recognition_model_path) or not detector_model_path or not os.path.exists(detector_model_path):
            self.run_inference_btn.setEnabled(False)
            self.inference_model = None
            self.detector_model = None
            self.class_names = None
            print(f"Model not found. Recognition: {recognition_model_path}, Detector: {detector_model_path}")
            return

        try:
            # Load both models
            self.detector_model = YOLO(detector_model_path)
            self.inference_model = keras.models.load_model(recognition_model_path)
            self.class_names = model_data.model_classes
            self.run_inference_btn.setEnabled(True)
            print("Inference and Detector models loaded successfully.")
        except Exception as e:
            self.run_inference_btn.setEnabled(False)
            QMessageBox.critical(self, "Erro ao Carregar Modelo", f"Não foi possível carregar o modelo de inferência:\n{e}")

    @Slot()
    def run_inference(self):
        """
        Executes the character detection and recognition process.
        """
        if not self.detector_model or not self.inference_model or not self.class_names:
            QMessageBox.warning(self, "Modelo não Carregado", "Nenhum modelo de inferência foi carregado.")
            return

        cam_index = self.camera_select.currentIndex()
        source_image = self.parent_wnd.cameras[cam_index].actual_image

        if source_image is None:
            QMessageBox.warning(self, "Sem Imagem", "Nenhuma imagem carregada na câmera selecionada.")
            return

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
        img_with_boxes = source_image.copy()
        model_data = self.parent_wnd.model_json.model
        img_h, img_w = model_data.image_height, model_data.image_width

        for box in character_boxes:
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
            input_data = np.expand_dims(np.expand_dims(normalized_char, axis=-1), axis=0)
            
            predicted_char = "?"
            confidence = 0.0
            prediction = self.inference_model.predict(input_data, verbose=0)
            if prediction.any():
                predicted_idx = np.argmax(prediction)
                predicted_char = self.class_names[predicted_idx]
                confidence = np.max(prediction) * 100
                predicted_string += predicted_char

            # Draw bounding box and label on the result image
            cv2.rectangle(img_with_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(img_with_boxes, predicted_char, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Add individual result to the list widget
            item = QListWidgetItem(self.char_list_widget)
            custom_widget = InferenceResultItem(
                resized_char, # Show the preprocessed image
                predicted_char,
                confidence
            )
            item.setSizeHint(QSize(150, 60))
            self.char_list_widget.addItem(item)
            self.char_list_widget.setItemWidget(item, custom_widget)

        # Update UI
        self.predicted_text_label.setText(f"<b>{predicted_string}</b>")

        # Compare with ground truth if available
        ground_truth = self.ground_truth_edit.text()
        if ground_truth:
            correct_chars = sum(1 for i in range(min(len(ground_truth), len(predicted_string))) if ground_truth[i] == predicted_string[i])
            accuracy = (correct_chars / len(ground_truth)) * 100 if len(ground_truth) > 0 else 0
            self.accuracy_label.setText(f"<b>{accuracy:.2f}%</b>")
        else:
            self.accuracy_label.setText("<b>N/A</b>")

        # Display the result image
        qt_image = vision_lib.convert_qt_image(img_with_boxes)
        self.result_image_label.setPixmap(qt_image.scaled(
            self.result_image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))