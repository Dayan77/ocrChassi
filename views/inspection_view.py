import sys
import os
import json
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QGridLayout,
    QSpacerItem,
    QSizePolicy,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
)
from PySide6.QtGui import QPixmap, QFont, QImage
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QObject
import cv2
import numpy as np
import tensorflow as tf
from pv_visionlib import pvVisionLib

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    import keras_ocr
except ImportError:
    keras_ocr = None

# Assuming config_ini.py is in the parent directory
sys.path.append("..")
import config_ini

try:
    import torch
    from components.models import SimpleCNN
except ImportError:
    torch = None
    SimpleCNN = None

class InferenceWorker(QObject):
    finished = Signal(str, list, list)

    def __init__(self, detector_model, recognition_model, model_data, library, frames_with_source, device=None):
        super().__init__()
        self.detector_model = detector_model
        self.recognition_model = recognition_model
        self.model_data = model_data
        self.library = library
        self.frames_with_source = frames_with_source  # list of tuples: [(frame, cam_index)]
        self.device = device

    def run(self):
        all_results = [] # list to hold final string outputs to emit per source
        annotated_frames_with_source = [] # list to hold (annotated_frame, cam_index)
        parsed_chars_by_source = [] # list of (cam_index, [(char, conf), ...])
        for frame, cam_index in self.frames_with_source:
            annotated_frame = frame.copy()
            predicted_string = ""
            chars_with_conf = []

            if self.library == "EasyOCR" and easyocr:
                reader = easyocr.Reader(['en'], gpu=True)
                results = reader.readtext(frame)
                for (bbox, text, prob) in results:
                    predicted_string += text + " "
                    chars_with_conf.append((text, float(prob)))
                    (tl, tr, br, bl) = bbox
                    tl = (int(tl[0]), int(tl[1]))
                    br = (int(br[0]), int(br[1]))
                    cv2.rectangle(annotated_frame, tl, br, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            elif self.library == "KerasOCR" and keras_ocr:
                pipeline = keras_ocr.pipeline.Pipeline()
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                prediction_groups = pipeline.recognize([img_rgb])
                for text, box in prediction_groups[0]:
                    predicted_string += text + " "
                    chars_with_conf.append((text, 1.0))
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    x1, y1 = int(min(xs)), int(min(ys))
                    x2, y2 = int(max(xs)), int(max(ys))
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            elif self.detector_model:
                # YOLO + Recognition
                results = self.detector_model(frame, verbose=False)
                boxes = results[0].boxes.xyxy.cpu().numpy()
                boxes = sorted(boxes, key=lambda x: x[0])
                
                img_h = int(self.model_data.get('image_height', 64))
                img_w = int(self.model_data.get('image_width', 64))
                class_names = self.model_data.get('model_classes', [])
                if not class_names:
                    print("Aviso: 'model_classes' está vazio ou não encontrado no JSON.")
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    h_img, w_img = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img, x2), min(h_img, y2)
                    
                    char_img = frame[y1:y2, x1:x2]
                    if char_img.size == 0: continue

                    gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
                    h, w = gray.shape
                    if h > w:
                        pad = (h - w) // 2
                        gray = cv2.copyMakeBorder(gray, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[0])
                    elif w > h:
                        pad = (w - h) // 2
                        gray = cv2.copyMakeBorder(gray, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0])
                    
                    resized = cv2.resize(gray, (img_w, img_h), interpolation=cv2.INTER_AREA)
                    normalized = resized / 255.0
                    
                    char_text = "?"
                    conf_val = 0.0
                    if self.library == "TensorFlow" and self.recognition_model:
                        input_data = np.expand_dims(normalized, axis=0)
                        input_data = np.expand_dims(input_data, axis=-1)
                        pred = self.recognition_model.predict(input_data, verbose=0)
                        idx = np.argmax(pred)
                        conf_val = float(np.max(pred))
                        if idx < len(class_names):
                            char_text = class_names[idx]
                        else:
                            print(f"Índice previsto {idx} fora dos limites (0-{len(class_names)-1})")
                    elif self.library == "PyTorch" and self.recognition_model and torch:
                        input_tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).float().to(self.device)
                        with torch.no_grad():
                            outputs = self.recognition_model(input_tensor)
                            probs = torch.softmax(outputs, dim=1)
                            conf, pred_idx = torch.max(probs, 1)
                            idx = pred_idx.item()
                            conf_val = float(conf.item())
                            if idx < len(class_names):
                                char_text = class_names[idx]
                            else:
                                print(f"Índice previsto {idx} fora dos limites (0-{len(class_names)-1})")
                    
                    predicted_string += char_text
                    chars_with_conf.append((char_text, conf_val))
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, char_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            result_label = f"Cam {cam_index + 1}: {predicted_string.strip()}"
            all_results.append(result_label)
            annotated_frames_with_source.append((annotated_frame, cam_index))
            parsed_chars_by_source.append((cam_index, chars_with_conf))
        
        # Join multiple result strings if both cameras fired in one manual inspection
        final_text_output = " | ".join(all_results)
        self.finished.emit(final_text_output, annotated_frames_with_source, parsed_chars_by_source)


class ZoomableLabel(QLabel):
    def __init__(self, text=""):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #000; border: 1px solid #45475a;")
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pixmap = None
        self._scale_factor = 1.0
        
        # Panning state
        self._pan_active = False
        self._pan_start_pos = None
        self._pan_offset_x = 0
        self._pan_offset_y = 0

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        # Reset scale factor and panning on new image
        self._scale_factor = 1.0
        self._pan_offset_x = 0
        self._pan_offset_y = 0
        self._update_view()

    def wheelEvent(self, event):
        if not self._pixmap:
            return
            
        # Delta is typically 120 per notch
        # We increase or decrease scale factor by 10%
        angle = event.angleDelta().y()
        if angle > 0:
            self._scale_factor *= 1.1  # Zoom in
        else:
            self._scale_factor *= 0.9  # Zoom out

        # Limit minimum scale so it doesn't vanish entirely
        if self._scale_factor < 0.1:
            self._scale_factor = 0.1
            
        self._update_view()

    def resizeEvent(self, event):
        if self._pixmap:
            self._update_view()
        super().resizeEvent(event)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._scale_factor > 1.0:
            self._pan_active = True
            self._pan_start_pos = event.position()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._pan_active and self._pan_start_pos:
            delta = event.position() - self._pan_start_pos
            self._pan_start_pos = event.position()
            
            self._pan_offset_x += delta.x()
            self._pan_offset_y += delta.y()
            self._update_view()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._pan_active = False
            self.setCursor(Qt.ArrowCursor)

    def _update_view(self):
        if not self._pixmap:
            return
            
        # Base scale to fit inside widget
        base_pixmap = self._pixmap.scaled(
            self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        # Apply user scale if zoomed
        final_w = int(base_pixmap.width() * self._scale_factor)
        final_h = int(base_pixmap.height() * self._scale_factor)
        
        zoomed_pixmap = base_pixmap.scaled(
            final_w, final_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        # If zoomed in, apply panning (crop the zoomed pixmap)
        if self._scale_factor > 1.0:
            # Calculate maximum allowed pan offsets to restrict panning past image edges
            max_offset_x = (zoomed_pixmap.width() - self.width()) / 2
            max_offset_y = (zoomed_pixmap.height() - self.height()) / 2
            
            if max_offset_x < 0: max_offset_x = 0
            if max_offset_y < 0: max_offset_y = 0
            
            # Constrain panic offset
            self._pan_offset_x = max(min(self._pan_offset_x, max_offset_x), -max_offset_x)
            self._pan_offset_y = max(min(self._pan_offset_y, max_offset_y), -max_offset_y)
            
            # Translate source rect according to pan offsets
            crop_x = int((zoomed_pixmap.width() - self.width()) / 2 - self._pan_offset_x)
            crop_y = int((zoomed_pixmap.height() - self.height()) / 2 - self._pan_offset_y)
            
            # Ensure crop boundaries don't go out of range
            crop_x = max(0, min(crop_x, zoomed_pixmap.width() - self.width()))
            crop_y = max(0, min(crop_y, zoomed_pixmap.height() - self.height()))
            
            final_pixmap = zoomed_pixmap.copy(crop_x, crop_y, self.width(), self.height())
        else:
            # If zoomed out or neutral, center it normally (offsets ignored)
            self._pan_offset_x = 0
            self._pan_offset_y = 0
            final_pixmap = zoomed_pixmap
            
        super().setPixmap(final_pixmap)


class InspectionView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Inspeção Automática")
        self.vision_lib = pvVisionLib()
        self.detector_model = None
        self.recognition_model = None
        self.model_data = {}
        self.library = "TensorFlow"
        self.device = None
        self.inspection_count = 0
        self.ocr_results = []
        
        # Cache for raw frames to be reused in Redo
        self.cam1_raw_frame = None
        self.cam2_raw_frame = None

        # Main layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # --- Left Panel (Cameras) ---
        left_card = QWidget()
        left_card.setProperty("class", "card")
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(20, 20, 20, 20)

        # Title for Cameras
        cam_title = QLabel("Câmeras e Resultados")
        cam_title.setProperty("class", "card_title")
        left_layout.addWidget(cam_title)

        # Camera Grid
        camera_layout = QGridLayout()
        
        # Camera 1
        cam1_container = QWidget()
        cam1_layout = QVBoxLayout(cam1_container)
        self.cam1_view = ZoomableLabel("CAM 1")
        self.load_image_btn1 = QPushButton("Load Image")
        cam1_layout.addWidget(self.cam1_view, 1)
        cam1_layout.addWidget(self.load_image_btn1, 0)
        
        # Camera 2
        cam2_container = QWidget()
        cam2_layout = QVBoxLayout(cam2_container)
        self.cam2_view = ZoomableLabel("CAM 2")
        self.load_image_btn2 = QPushButton("Load Image")
        cam2_layout.addWidget(self.cam2_view, 1)
        cam2_layout.addWidget(self.load_image_btn2, 0)

        camera_layout.addWidget(cam1_container, 0, 0)
        camera_layout.addWidget(cam2_container, 0, 1)
        left_layout.addLayout(camera_layout, 1) # Set stretch to max for cameras

        # Results display (Inside Left Card)
        results_widget = QWidget()
        results_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        results_layout = QVBoxLayout(results_widget)
        self.results_label = QLabel("Resultado da Inspeção:")
        self.results_label.setFont(QFont("Arial", 16))
        
        self.results_display = QLabel("N/A")
        self.results_display.setFont(QFont("Arial", 72, QFont.Bold))
        self.results_display.setAlignment(Qt.AlignCenter)
        self.results_display.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.results_display.setMinimumHeight(120)
        self.results_display.setStyleSheet(
            "border: 2px solid #313244; padding: 10px; background-color: #1e1e2e; color: #89b4fa; border-radius: 6px; font-size: 72pt; font-weight: bold;"
        )
        results_layout.addWidget(self.results_label)
        results_layout.addWidget(self.results_display)
        
        left_layout.addWidget(results_widget, 0)


        # --- Right Panel (Controls) ---
        right_card = QWidget()
        right_card.setProperty("class", "card")
        right_panel = QVBoxLayout(right_card)
        right_panel.setContentsMargins(20, 20, 20, 20)
        right_panel.setSpacing(15)

        ctrl_title = QLabel("Controles")
        ctrl_title.setProperty("class", "card_title")
        
        # Add Title and Redo Button in an HBox
        ctrl_header_layout = QHBoxLayout()
        ctrl_header_layout.addWidget(ctrl_title)
        ctrl_header_layout.addStretch()
        
        self.redo_button = QPushButton("Refazer Inspeção")
        self.redo_button.setStyleSheet("""
            QPushButton {
                background-color: #a6e3a1; 
                color: #11111b; 
                border-radius: 4px; 
                padding: 5px 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #94e2d5;
            }
        """)
        ctrl_header_layout.addWidget(self.redo_button)
        
        right_panel.addLayout(ctrl_header_layout)
        
        self.loaded_model_label = QLabel("Modelo: Nenhum")
        self.loaded_model_label.setStyleSheet("color: #a6adc8; font-weight: bold; margin-bottom: 5px;")
        right_panel.addWidget(self.loaded_model_label)

        self.load_model_button = QPushButton("Carregar Modelo")
        
        self.manual_inspection_button = QPushButton("Inspeção Manual")
        self.manual_inspection_button.setProperty("class", "primary")
        
        self.continuous_inspection_button = QPushButton("Iniciar Inspeção Contínua")
        # Default styling
        
        self.inspection_counter_label = QLabel(f"Inspeções: {self.inspection_count}")
        self.inspection_counter_label.setStyleSheet("font-size: 14px; color: #a6adc8; margin-top: 10px;")
        
        self.ocr_results_list = QListWidget()


        right_panel.addWidget(self.load_model_button)
        right_panel.addWidget(self.manual_inspection_button)
        right_panel.addWidget(self.continuous_inspection_button)
        right_panel.addWidget(self.inspection_counter_label)
        right_panel.addWidget(self.ocr_results_list)
        right_panel.addSpacerItem(
            QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

        # Add cards to main layout
        main_layout.addWidget(left_card, 70) # 70% width
        main_layout.addWidget(right_card, 30) # 30% width

        self.load_stylesheet()

        # Camera Initialization
        self.caps = []
        for i in range(config_ini.cam_qty):
            cap = cv2.VideoCapture(config_ini.cam_usb_index[i])
            if cap.isOpened():
                self.caps.append(cap)
            else:
                print(f"Erro ao abrir a câmera {i}")
                self.caps.append(None)

        # Timer for continuous inspection
        self.inspection_timer = QTimer(self)
        self.inspection_timer.setInterval(config_ini.trigger_delay)
        self.inspection_timer.timeout.connect(self.run_manual_inspection)

        # Connect buttons to functions
        self.load_model_button.clicked.connect(self.load_model)
        self.manual_inspection_button.clicked.connect(self.run_manual_inspection)
        self.continuous_inspection_button.clicked.connect(
            self.toggle_continuous_inspection
        )
        self.load_image_btn1.clicked.connect(lambda: self.load_image_for_view(0))
        self.load_image_btn2.clicked.connect(lambda: self.load_image_for_view(1))
        self.redo_button.clicked.connect(self.redo_inference)

    def redo_inference(self):
        # Clear main screen result box, keep history intact
        self.results_display.setText("N/A")
        
        frames_with_source = []
        
        # Use raw un-annotated frames
        if self.cam1_raw_frame is not None:
            frames_with_source.append((self.cam1_raw_frame.copy(), 0))
            
        if self.cam2_raw_frame is not None:
            frames_with_source.append((self.cam2_raw_frame.copy(), 1))

        if not frames_with_source:
            print("Nenhuma imagem original carregada para refazer a inspeção.")
            return
            
        self.run_inference_on_frames(frames_with_source)

    def shutdown(self):
        self.inspection_timer.stop()
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        for cap in self.caps:
            if cap:
                cap.release()

    def load_stylesheet(self):
        try:
            with open("style.qss", "r") as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            print("style.qss not found. Using default styles.")

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar Configuração do Modelo",
            "",
            "JSON Files (*.json)"
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.model_data = json.load(f)
                
                # Handle potential nested 'model' key
                if 'model' in self.model_data and isinstance(self.model_data['model'], dict):
                    self.model_data = self.model_data['model']
                
                print(f"Configuração carregada. Classes: {len(self.model_data.get('model_classes', []))}")
                
                if hasattr(config_ini, 'production_model_library'):
                    self.library = config_ini.production_model_library
                else:
                    self.library = self.model_data.get('production_model_library', 'TensorFlow')
                
                # Load Detector
                detector_path = self.model_data.get('detector_model_path')
                if detector_path and os.path.exists(detector_path):
                    if YOLO:
                        self.detector_model = YOLO(detector_path)
                    else:
                        print("YOLO not installed.")
                
                # Load Recognition
                if self.library == 'TensorFlow':
                    rec_path = self.model_data.get('encoder_filename')
                    if rec_path and os.path.exists(rec_path):
                        self.recognition_model = tf.keras.models.load_model(rec_path, compile=False)
                elif self.library == 'PyTorch':
                    rec_path = self.model_data.get('encoder_filename')
                    if rec_path:
                        base, _ = os.path.splitext(rec_path)
                        rec_path = base + ".pth"
                    
                    if rec_path and os.path.exists(rec_path):
                        if torch and SimpleCNN:
                            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            num_classes = len(self.model_data.get('model_classes', []))
                            h = int(self.model_data.get('image_height', 64))
                            w = int(self.model_data.get('image_width', 64))
                            self.recognition_model = SimpleCNN(num_classes, h, w).to(self.device)
                            self.recognition_model.load_state_dict(torch.load(rec_path, map_location=self.device))
                            self.recognition_model.eval()

                model_name = self.model_data.get('model_name', os.path.basename(file_path))
                self.setWindowTitle(f"Inspeção Automática - {model_name}")
                self.loaded_model_label.setText(f"Modelo: {model_name}")
                print("Modelo carregado com sucesso!")
            except Exception as e:
                print(f"Erro ao carregar o modelo: {e}")

    def load_image_for_view(self, view_index):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.png *.jpg *.bmp)"
        )
        if file_path:
            frame = cv2.imread(file_path)
            if frame is not None:
                if view_index == 0 and getattr(config_ini, 'production_cam1_flip', False):
                    frame = cv2.flip(frame, 1)
                elif view_index == 1 and getattr(config_ini, 'production_cam2_flip', False):
                    frame = cv2.flip(frame, 1)

                pixmap = self.convert_cv_to_pixmap(frame)
                if view_index == 0:
                    self.cam1_raw_frame = frame.copy()
                    self.update_camera_view(self.cam1_view, pixmap)
                else:
                    self.cam2_raw_frame = frame.copy()
                    self.update_camera_view(self.cam2_view, pixmap)
                
                self.run_inference_on_frames([(frame, view_index)])
                
    def run_inference_on_frames(self, frames_with_source):
        if not self.detector_model and self.library not in ["EasyOCR", "KerasOCR"]:
            print("Modelo não carregado. Carregue um modelo primeiro.")
            return

        # Create and start the inference thread
        self.thread = QThread()
        self.worker = InferenceWorker(self.detector_model, self.recognition_model, self.model_data, self.library, frames_with_source, self.device)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.update_results)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def run_manual_inspection(self):
        # Get frames from cameras
        frames_with_source = []
        for i, cap in enumerate(self.caps):
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    if i == 0 and getattr(config_ini, 'production_cam1_flip', False):
                        frame = cv2.flip(frame, 1)
                    elif i == 1 and getattr(config_ini, 'production_cam2_flip', False):
                        frame = cv2.flip(frame, 1)

                    frames_with_source.append((frame.copy(), i))
                    pixmap = self.convert_cv_to_pixmap(frame)
                    if i == 0:
                        self.cam1_raw_frame = frame.copy()
                        self.update_camera_view(self.cam1_view, pixmap)
                    elif i == 1:
                        self.cam2_raw_frame = frame.copy()
                        self.update_camera_view(self.cam2_view, pixmap)
            else:
                print(f"Erro ao abrir a câmera {i}")

        if not frames_with_source:
            print("Não foi possível capturar imagens das câmeras.")
            return
        
        self.run_inference_on_frames(frames_with_source)

    def update_results(self, characters_output, annotated_frames_with_source, parsed_chars_by_source):
        # We process parsed_chars_by_source to build the final combined string
        parsed_chars_by_source.sort(key=lambda x: x[0])
        
        cam1_chars = []
        cam2_chars = []
        for cam_idx, chars in parsed_chars_by_source:
             if cam_idx == 0: cam1_chars.extend(chars)
             if cam_idx == 1: cam2_chars.extend(chars)
             
        target_len = getattr(config_ini, 'number_of_characters', 10)
        
        # Merge logic to find best split for Cam1 prefix and Cam2 suffix
        if not cam1_chars and not cam2_chars:
             final_string = characters_output
        elif not cam1_chars:
             final_string = "".join(c for c, conf in cam2_chars)
        elif not cam2_chars:
             final_string = "".join(c for c, conf in cam1_chars)
        else:
            L1 = len(cam1_chars)
            L2 = len(cam2_chars)
            # If total detected chars is less than or equal to target, concatenate all
            if L1 + L2 <= target_len:
                final_string = "".join(c for c, conf in cam1_chars) + "".join(c for c, conf in cam2_chars)
            else:
                best_sum = -1.0
                best_split = (L1, target_len - L1) if L1 <= target_len else (target_len, 0)
                
                # Iterate over possible k1 (number of chars to take from Cam1)
                # k1 must be <= L1, and k2 = target_len - k1 must be <= L2
                min_k1 = max(0, target_len - L2)
                max_k1 = min(target_len, L1)
                
                for k1 in range(min_k1, max_k1 + 1):
                    k2 = target_len - k1
                    
                    # Sum probabilities of chosen segments
                    sum1 = sum(conf for c, conf in cam1_chars[:k1]) if k1 > 0 else 0
                    sum2 = sum(conf for c, conf in cam2_chars[L2-k2:]) if k2 > 0 else 0
                    total = sum1 + sum2
                    
                    if total > best_sum:
                        best_sum = total
                        best_split = (k1, k2)
                        
                k1, k2 = best_split
                part1 = "".join(c for c, conf in cam1_chars[:k1])
                part2 = "".join(c for c, conf in cam2_chars[L2-k2:]) if k2 > 0 else ""
                final_string = part1 + part2

        display_text = final_string if final_string else characters_output
        if not display_text.strip():
            display_text = "N/A"
            characters_output = "N/A"
        self.results_display.setText(display_text)
        self.inspection_count += 1
        self.inspection_counter_label.setText(f"Inspeções: {self.inspection_count}")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result_text = f"{timestamp} - {display_text}  ({characters_output})"
        
        self.ocr_results.insert(0, result_text)
        self.ocr_results_list.insertItem(0, result_text)
        
        # Optional: Limit the list size to e.g. 100 entries
        if len(self.ocr_results) > 100:
            self.ocr_results.pop()
            self.ocr_results_list.takeItem(100)
            
        for frame, source in annotated_frames_with_source:
            pixmap = self.convert_cv_to_pixmap(frame)
            if source == 0:
                self.update_camera_view(self.cam1_view, pixmap)
            elif source == 1:
                self.update_camera_view(self.cam2_view, pixmap)

    def toggle_continuous_inspection(self):
        if self.inspection_timer.isActive():
            self.inspection_timer.stop()
            self.continuous_inspection_button.setText("Iniciar Inspeção Contínua")
            self.continuous_inspection_button.setProperty("class", "") # Reset to default
            self.continuous_inspection_button.style().unpolish(self.continuous_inspection_button)
            self.continuous_inspection_button.style().polish(self.continuous_inspection_button)
            print("Inspeção contínua parada.")
        else:
            self.inspection_timer.start()
            self.continuous_inspection_button.setText("Parar Inspeção Contínua")
            self.continuous_inspection_button.setProperty("class", "danger") # Set to danger (Red)
            self.continuous_inspection_button.style().unpolish(self.continuous_inspection_button)
            self.continuous_inspection_button.style().polish(self.continuous_inspection_button)
            print("Inspeção contínua iniciada.")

    def update_camera_view(self, label, pixmap):
        # The ZoomableLabel class handles the scaling now
        label.setPixmap(pixmap)

    def convert_cv_to_pixmap(self, cv_img):
        """Convert from an opencv image to a QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        # Needs to be Format_RGB888 to match earlier conversion from BGR -> RGB
        convert_to_Qt_format = QImage(
            rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888
        )
        return QPixmap.fromImage(convert_to_Qt_format).copy() # return isolated copy