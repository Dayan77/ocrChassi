import sys
import qtawesome as qta
from PySide6.QtCore import Qt, QSize, Signal, Slot
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QLabel, QListWidget, QListWidgetItem, QPushButton, QMessageBox,
    QScrollArea,
    QSizePolicy, QGraphicsDropShadowEffect, QFrame,
    QCheckBox, QComboBox, QDateEdit, QDateTimeEdit, QDial, QDoubleSpinBox,
    QFontComboBox, QLCDNumber, QLineEdit, QProgressBar, QRadioButton, QSlider,
    QSpinBox, QTimeEdit, QGroupBox, QTabWidget, QFormLayout, QFileDialog
)

from pathlib import Path

from PySide6.QtGui import QColor, QIcon, QPixmap
from qt_material import apply_stylesheet

import pv_visionlib
from components.datasetview import DatasetView, prepare_yolo_dataset, prepare_recognition_dataset
from components.trainingsummaryview import TrainingSummaryView
from components.trainingprocess import TrainingProcessDialog
from components.resultsview import ResultsView
from components.inferenceview import InferenceView
from components.detectortrainingdialog import DetectorTrainingDialog


class CardModel(QWidget):

    def __init__(self, parent=None, color_bg="#2c3e50"):
        super().__init__(parent)
        self.parent_wnd = parent
        # Set a fixed background color for the card
        self.setStyleSheet(f"background-color: {color_bg}; border-radius: 10px;")
        
        # Create a layout for the card's content
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(1, 1, 1, 1)
        self.layout.setSpacing(1)
        
        # Apply a drop shadow effect
        self.shadow_effect = QGraphicsDropShadowEffect(self)
        self.shadow_effect.setBlurRadius(20)
        self.shadow_effect.setXOffset(0)
        self.shadow_effect.setYOffset(5)
        self.shadow_effect.setColor(QColor(0, 0, 0, 100)) # Semi-transparent black
        self.setGraphicsEffect(self.shadow_effect)
        
    def set_content(self, widget):
        """Adds a widget to the card's layout."""
        self.layout.addWidget(widget)



# Custom widget to display inside the card
class TrainingCardContent(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.parent_wnd = parent
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        self.title_label = QLabel("Card Title")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.title_label.setMinimumHeight(60)
        self.title_label.setMaximumHeight(60)
        #layout.addWidget(self.title_label)
        
        # Separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        #layout.addWidget(line)
        
        
        self.model_train_box = ModelView(self.parent_wnd)
        #image_segmentation_box.setStyleSheet("font-size: 12px; background-color: #536276ff")
        self.model_train_box.setMinimumHeight(350)
        self.model_train_box.setMaximumHeight(550)
        self.model_train_box.setMinimumWidth(500)
        self.model_train_box.setMaximumWidth(800)
        layout.addWidget(self.model_train_box)


        #test_button = QPushButton
        test_button = QPushButton()
        test_button.setMaximumHeight(35)
        test_button.setMaximumWidth(60)
        test_icon = QIcon(":icons/icons/grid.svg") 
        test_button.setIcon(test_icon)

        # Style the button to be partially transparent and smaller
        test_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 10); /* Semi-transparent black */
                color: white;                       
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)

        #layout.addWidget(test_button)
        test_button.clicked.connect(self.test_pytesseract)

    def test_pytesseract(self):
        visionLib = pv_visionlib.pvVisionLib()
        visionLib.collect_samples(self.parent_wnd.cameras[0].image_path, '../models/', 'ocr_samples')

        

class ModelView(QFrame):

    jsonLoaded = Signal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_wnd = parent
        height = 440
        self.tabs = QTabWidget(self)
        self.tabs.setFixedHeight(height)
        self.tabs.setFixedWidth(800)
        
        self.tab_config = QWidget()
        self.tab_config.setFixedHeight(height)
        self.tab_config.setFixedWidth(800)
        self.tab_edit = QWidget()
        self.tab_edit.setFixedHeight(height)
        self.tab_edit.setFixedWidth(800)
        self.tab_data = QWidget()
        self.tab_data.setFixedHeight(height)
        self.tab_data.setFixedWidth(800)
        self.tab_results = QWidget()
        self.tab_results.setFixedHeight(height)
        self.tab_results.setFixedWidth(800)
        self.tab_inference = QWidget()
        self.tab_inference.setFixedHeight(height)
        self.tab_inference.setFixedWidth(800)

        self.tabs.addTab(self.tab_config, "Modelo")
        self.tabs.addTab(self.tab_edit, "Anotações")
        self.tabs.addTab(self.tab_data, "Treinamento")
        self.tabs.addTab(self.tab_results, "Resultados")
        self.tabs.addTab(self.tab_inference, "Inferência")

        # Pass self.parent_wnd (ProgramView) to ModelJsonView so it can access the model_json object
        self.json_model_view = ModelJsonView(self.parent_wnd)
        # Pass the signal up from the child view
        self.json_model_view.jsonLoaded.connect(self.jsonLoaded)

        # --- Integrate DatasetView into the "Anotações" tab ---
        self.dataset_view = DatasetView()
        edit_tab_layout = QVBoxLayout(self.tab_edit)
        edit_tab_layout.setContentsMargins(0, 0, 0, 0)
        edit_tab_layout.addWidget(self.dataset_view)

        # --- Integrate TrainingSummaryView into the "Treinamento" tab ---
        self.training_summary_view = TrainingSummaryView()
        train_tab_layout = QVBoxLayout(self.tab_data)
        train_tab_layout.addWidget(self.training_summary_view)
        self.dataset_view.datasetLoaded.connect(self.update_training_summary)
        self.training_summary_view.startTrainingClicked.connect(self.start_training)
        self.training_summary_view.prepareRecognitionDataClicked.connect(self.on_prepare_recognition_data_clicked)
        self.training_summary_view.prepareYoloDataClicked.connect(self.on_prepare_yolo_data_clicked)
        self.training_summary_view.startDetectorTrainingClicked.connect(self.start_detector_training)

        # --- Integrate ResultsView into the "Resultados" tab ---
        self.results_view = ResultsView()
        results_tab_layout = QVBoxLayout(self.tab_results)
        results_tab_layout.addWidget(self.results_view)

        # --- Integrate InferenceView into the "Inferência" tab ---
        self.inference_view = InferenceView(self.parent_wnd)
        inference_tab_layout = QVBoxLayout(self.tab_inference)
        inference_tab_layout.addWidget(self.inference_view)
        # Connect a signal to notify InferenceView when a model is loaded/changed
        self.parent_wnd.modelDataChanged.connect(self.inference_view.on_model_loaded)
        self.inference_view.on_model_loaded() # Call once at startup

        # --- Setup for scrollable config tab ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setWidget(self.json_model_view)

        config_tab_layout = QVBoxLayout(self.tab_config)
        config_tab_layout.setContentsMargins(0, 0, 0, 0)
        config_tab_layout.addWidget(scroll_area)

        # This is no longer needed as we use a layout with a scroll area
        # self.tab_config.setLayout(self.json_model_view)

    def update_all_views(self):
        """Update all child views with the new model data."""
        self.json_model_view.update_json_values()
        self.on_model_data_changed() # This will trigger dataset view to load


    @Slot()
    def on_model_data_changed(self):
        """Triggered after ProgramView loads new model data."""
        if self.parent_wnd and self.parent_wnd.model_json.model:
            train_path = self.parent_wnd.model_json.model.model_train_dataset
            self.dataset_view.load_dataset(train_path)

    def update_training_summary(self, dataset_summary):
        """Slot to update the training summary view."""
        model_data = self.parent_wnd.model_json.model
        self.training_summary_view.update_summary(model_data, dataset_summary)

    def start_training(self):
        """Slot to initiate the training process."""
        model_data = self.parent_wnd.model_json.model
        if not model_data:
            # You might want to show a QMessageBox here
            print("No model data loaded.")
            return
        
        dialog = TrainingProcessDialog(model_data, self)
        dialog.trainingCompleted.connect(self.on_training_completed)
        dialog.validationTestCompleted.connect(self.on_validation_completed)
        dialog.exec()

    def start_detector_training(self):
        """Slot to initiate the YOLO detector training process."""
        model_data = self.parent_wnd.model_json.model
        if not model_data:
            QMessageBox.warning(self, "No Model Data", "Please load a model configuration first.")
            return

        dialog = DetectorTrainingDialog(model_data, self)
        dialog.exec()

    @Slot()
    def on_prepare_yolo_data_clicked(self):
        """Slot to handle the 'Prepare YOLO Data' button click."""
        model_data = self.parent_wnd.model_json.model
        if not model_data:
            QMessageBox.warning(self, "No Model Data", "Please load a model configuration first.")
            return

        source_path = model_data.annotation_dataset_path
        destination_path = model_data.yolo_dataset_path
        
        prepare_yolo_dataset(source_path, destination_path, parent_widget=self)

    @Slot()
    def on_prepare_recognition_data_clicked(self):
        """Slot to handle the 'Prepare Recognition Data' button click."""
        model_data = self.parent_wnd.model_json.model
        if not model_data:
            QMessageBox.warning(self, "No Model Data", "Please load a model configuration first.")
            return

        source_path = model_data.annotation_dataset_path
        destination_path = model_data.model_train_dataset
        detector_path = model_data.detector_model_path

        prepare_recognition_dataset(source_path, destination_path, detector_path, model_data, parent_widget=self)

    @Slot(object)
    def on_training_completed(self, result):
        """Receives results from the training dialog and updates relevant views."""
        stats, class_names = result

        # Update the results view with stats
        self.results_view.update_training_summary(stats)

        # If training was successful and we have class names, update the model
        if stats and class_names:
            # Store the classes as a list, not a concatenated string
            self.parent_wnd.model_json.model.model_classes = class_names
            
            # Save the updated model back to the JSON file
            self.parent_wnd.model_json.save_to_file(self.parent_wnd.model_json.model.model_filename)
            
            # Update the UI in the "Modelo" tab to show the new class string
            self.json_model_view.update_json_values()
            print(f"Updated model classes to: {class_names} and saved to file.")

    @Slot(list)
    def on_validation_completed(self, results):
        """Receives validation results and passes them to the results view."""
        self.results_view.update_validation_results(results)

class ModelJsonView(QWidget):
    jsonLoaded = Signal(str)

    def __init__(self, parent_wnd = None):
        super().__init__()
        self.parent_wnd = parent_wnd

        layout = QVBoxLayout(self)
        
        #buttons
        json_actions_row = QHBoxLayout()
        
        open_json_btn = QPushButton("Carregar")
        open_json_btn.setMaximumHeight(35)
        open_json_btn.setMaximumWidth(100)
        open_json_icon = QIcon(":icons/icons/folder.svg") 
        open_json_btn.setIcon(open_json_icon)

        # Style the button to be partially transparent and smaller
        open_json_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 10); /* Semi-transparent black */
                color: white;                       
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)

        open_json_btn.clicked.connect(self.open_file_json)
        
        save_json_btn = QPushButton("Salvar")
        save_json_btn.setMaximumHeight(35)
        save_json_btn.setMaximumWidth(100)
        save_json_icon = QIcon(":icons/icons/save.svg") 
        save_json_btn.setIcon(save_json_icon)

        # Style the button to be partially transparent and smaller
        save_json_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 10); /* Semi-transparent black */
                color: white;                       
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        save_json_btn.clicked.connect(self.save_file_json)

        json_actions_row.addWidget(open_json_btn)
        json_actions_row.addWidget(save_json_btn)

        layout.addLayout(json_actions_row)

        #Form
        json_group_box = QGroupBox("Informações do modelo IA")
        # Style the QGroupBox to be partially transparent and smaller
        json_group_box.setStyleSheet("""
            QGroupBox {
                background-color: rgba(0, 0, 0, 10); /* Semi-transparent black */
                color: white;                       
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        json_form_layout = QFormLayout()
        json_form_layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        json_form_layout.setContentsMargins(0, 25, 0, 0)
        

        self.json_modelname_edit = QLineEdit()
        self.json_modelname_edit.setFixedWidth(250)
        # Style the QLineEddit to be partially transparent and smaller
        self.json_modelname_edit.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 10); /* Semi-transparent black */
                color: white;                       
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        json_form_layout.addRow("Modelo:", self.json_modelname_edit)

        json_filename_row = QHBoxLayout()
        self.json_filename_edit = QLineEdit()
        self.json_filename_edit.setFixedWidth(520)
        # Style the QLineEddit to be partially transparent and smaller
        self.json_filename_edit.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 10); /* Semi-transparent black */
                color: white;                       
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        json_filename_btn = QPushButton()
        filename_icon = QIcon(":icons/icons/external-link.svg") 
        json_filename_btn.setIcon(filename_icon)
        json_filename_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 10); /* Semi-transparent black */
                color: white;                       
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        json_filename_btn.clicked.connect(self.browse_filename)        

        json_filename_row.addWidget(self.json_filename_edit)
        json_filename_row.addWidget(json_filename_btn)

        json_form_layout.addRow("Arquivo:", json_filename_row)

        json_encoder_row = QHBoxLayout()
        self.json_encoderfile_edit = QLineEdit()
        self.json_encoderfile_edit.setFixedWidth(520)
        # Style the QLineEddit to be partially transparent and smaller
        self.json_encoderfile_edit.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 10); /* Semi-transparent black */
                color: white;                       
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)

        json_encoder_btn = QPushButton()
        encoder_icon = QIcon(":icons/icons/external-link.svg") 
        json_encoder_btn.setIcon(encoder_icon)
        json_encoder_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 10); /* Semi-transparent black */
                color: white;                       
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        json_encoder_btn.clicked.connect(self.browse_encoderfile)

        json_encoder_row.addWidget(self.json_encoderfile_edit)
        json_encoder_row.addWidget(json_encoder_btn)
        json_form_layout.addRow("Codificador:", json_encoder_row)

        json_traindataset_row = QHBoxLayout()
        self.json_traindataset_edit = QLineEdit()
        self.json_traindataset_edit.setFixedWidth(520)
        # Style the QLineEddit to be partially transparent and smaller
        self.json_traindataset_edit.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 10); /* Semi-transparent black */
                color: white;                       
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)

        json_traindataset_btn = QPushButton()
        traindataset_icon = QIcon(":icons/icons/external-link.svg") 
        json_traindataset_btn.setIcon(traindataset_icon)
        json_traindataset_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 10); /* Semi-transparent black */
                color: white;                       
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        json_traindataset_btn.clicked.connect(self.browse_train_folder)

        json_traindataset_row.addWidget(self.json_traindataset_edit)
        json_traindataset_row.addWidget(json_traindataset_btn)
        json_form_layout.addRow("Treinamento:", json_traindataset_row)

        json_testdataset_row = QHBoxLayout()
        self.json_testdataset_edit = QLineEdit()
        self.json_testdataset_edit.setFixedWidth(520)
        
        # Style the QLineEddit to be partially transparent and smaller
        self.json_testdataset_edit.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 10); /* Semi-transparent black */
                color: white;                       
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)

        json_testdataset_btn = QPushButton()
        json_testdataset_icon = QIcon(":icons/icons/external-link.svg")
        json_testdataset_btn.setIcon(json_testdataset_icon)
        json_testdataset_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 10); /* Semi-transparent black */
                color: white;                       
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        json_testdataset_btn.clicked.connect(self.browse_test_folder)

        json_testdataset_row.addWidget(self.json_testdataset_edit)
        json_testdataset_row.addWidget(json_testdataset_btn)

        json_form_layout.addRow("Validação:", json_testdataset_row)

        # --- YOLO Dataset Path ---
        yolo_dataset_row = QHBoxLayout()
        self.json_yolodataset_edit = QLineEdit()
        self.json_yolodataset_edit.setFixedWidth(520)
        self.json_yolodataset_edit.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 10);
                color: white;
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        yolo_dataset_btn = QPushButton()
        yolo_dataset_icon = QIcon(":icons/icons/crosshair.svg")
        yolo_dataset_btn.setIcon(yolo_dataset_icon)
        yolo_dataset_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 10);
                color: white;
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        yolo_dataset_btn.clicked.connect(self.browse_yolo_folder)
        yolo_dataset_row.addWidget(self.json_yolodataset_edit)
        yolo_dataset_row.addWidget(yolo_dataset_btn)
        json_form_layout.addRow("Detector Dataset (YOLO):", yolo_dataset_row)

        # --- Detector Model Path ---
        detector_model_row = QHBoxLayout()
        self.json_detectormodel_edit = QLineEdit()
        self.json_detectormodel_edit.setFixedWidth(520)
        self.json_detectormodel_edit.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 10);
                color: white;
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        detector_model_btn = QPushButton()
        detector_model_icon = QIcon(":icons/icons/cpu.svg")
        detector_model_btn.setIcon(detector_model_icon)
        detector_model_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 10);
                color: white;
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        detector_model_btn.clicked.connect(self.browse_detector_model)
        detector_model_row.addWidget(self.json_detectormodel_edit)
        detector_model_row.addWidget(detector_model_btn)
        json_form_layout.addRow("Detector Model:", detector_model_row)

        # --- Annotation Dataset Path ---
        annotation_dataset_row = QHBoxLayout()
        self.json_annotationdataset_edit = QLineEdit()
        self.json_annotationdataset_edit.setFixedWidth(520)
        self.json_annotationdataset_edit.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 10);
                color: white;
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        annotation_dataset_btn = QPushButton()
        annotation_dataset_icon = QIcon(":icons/icons/edit.svg")
        annotation_dataset_btn.setIcon(annotation_dataset_icon)
        annotation_dataset_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 10);
                color: white;
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        annotation_dataset_btn.clicked.connect(self.browse_annotation_folder)
        annotation_dataset_row.addWidget(self.json_annotationdataset_edit)
        annotation_dataset_row.addWidget(annotation_dataset_btn)
        json_form_layout.addRow("Annotation Dataset:", annotation_dataset_row)

        json_bottom_data_widget = QWidget()
        json_bottom_data_widget.setFixedHeight(80)
        json_bottom_data = QHBoxLayout()
 
        self.json_classes_edit = QLineEdit()
        #self.json_classes_edit.setFixedHeight(70)
        # Style the QLineEddit to be partially transparent and smaller
        self.json_classes_edit.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 10); /* Semi-transparent black */
                color: white;                       
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        #json_form_layout.addRow("Classes:", self.json_classes_edit)

        self.json_epochs_edit = QSpinBox()
        self.json_epochs_edit.setRange(1, 1000)
        # Style the QLineEddit to be partially transparent and smaller
        self.json_epochs_edit.setStyleSheet("""
            QSpinBox {
                background-color: rgba(0, 0, 0, 10); /* Semi-transparent black */
                color: white;                       
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        json_bottom_data.addWidget(self.json_classes_edit)
        #json_form_layout.addRow("Epocas:", self.json_epochs_edit)

        json_heights_data = QVBoxLayout()
        self.json_imageheight_edit = QSpinBox()
        self.json_imageheight_edit.setRange(1, 1000)
        #json_form_layout.addRow("Altura:", self.json_imageheight_edit)
        json_heights_data.addWidget(self.json_imageheight_edit)

        self.json_imagewidth_edit = QSpinBox()
        self.json_imagewidth_edit.setRange(1, 1000)
        #json_form_layout.addRow("Comprimento:", self.json_imagewidth_edit)
        json_heights_data.addWidget(self.json_imagewidth_edit)

        json_bottom_data.addLayout(json_heights_data)
        json_bottom_data.addWidget(self.json_epochs_edit)

        json_bottom_data_widget.setLayout(json_bottom_data)
        json_form_layout.addRow(json_bottom_data_widget)

        json_group_box.setLayout(json_form_layout)

        layout.addWidget(json_group_box)

    

    def save_file_json(self):
        if not self.parent_wnd.model_json:
            return False
        
        if self.json_modelname_edit.text() == "" or self.json_encoderfile_edit.text() == "" or self.json_filename_edit.text() == "":
            return False
        
        try:
            if not self.parent_wnd.model_json.model:
                self.parent_wnd.model_json.create_model(
                    self.json_modelname_edit.text(),
                    self.json_filename_edit.text(),
                    self.json_encoderfile_edit.text(), 
                    self.json_traindataset_edit.text(),
                    self.json_testdataset_edit.text(),
                    self.json_yolodataset_edit.text(),
                    self.json_annotationdataset_edit.text(),
                    self.json_detectormodel_edit.text(),
                    list(self.json_classes_edit.text()), # Convert string back to list of chars
                    self.json_epochs_edit.value(),
                    self.json_imageheight_edit.value(),
                    self.json_imagewidth_edit.value()
                )
            else:
                self.parent_wnd.model_json.model.model_name = self.json_modelname_edit.text()
                self.parent_wnd.model_json.model.model_filename = self.json_filename_edit.text()
                self.parent_wnd.model_json.model.encoder_filename =  self.json_encoderfile_edit.text() 
                self.parent_wnd.model_json.model.model_train_dataset =  self.json_traindataset_edit.text()
                self.parent_wnd.model_json.model.model_test_dataset =  self.json_testdataset_edit.text()
                self.parent_wnd.model_json.model.annotation_dataset_path = self.json_annotationdataset_edit.text()
                self.parent_wnd.model_json.model.detector_model_path = self.json_detectormodel_edit.text()
                self.parent_wnd.model_json.model.yolo_dataset_path = self.json_yolodataset_edit.text()
                self.parent_wnd.model_json.model.model_classes = list(self.json_classes_edit.text())
                self.parent_wnd.model_json.model.train_epochs = self.json_epochs_edit.value()
                self.parent_wnd.model_json.model.image_height = self.json_imageheight_edit.value()
                self.parent_wnd.model_json.model.image_width = self.json_imagewidth_edit.value()

            if self.parent_wnd.model_json.save_to_file(self.json_filename_edit.text()):
                QMessageBox.information(self, "Sucesso", "Arquivo JSON salvo com sucesso!")
                return True
            else:
                QMessageBox.warning(self, "Falha ao Salvar", "Não foi possível salvar o arquivo JSON.")
                return False
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Ocorreu um erro inesperado ao salvar:\n{e}")
            return False


    def open_file_any(self):
        filename = None
        dialog = QFileDialog()
        
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("All (*.* )")
        dialog.setViewMode(QFileDialog.ViewMode.List)
        
        filename, ok = dialog.getOpenFileName()
           
        return filename

    def open_dir(self):
        dir_name = QFileDialog.getExistingDirectory(caption="Selecione um diretorio")
        if dir_name:
            return Path(dir_name)
        return None
    
    def browse_filename(self):
        filename = self.open_file_json()
        self.json_filename_edit.setText(filename)

    def browse_encoderfile(self):
        filename = self.open_file_any()
        self.json_encoderfile_edit.setText(filename)

    def browse_train_folder(self):
        dir = self.open_dir()
        if not dir:
            return
        self.json_traindataset_edit.setText(dir.as_posix())

    def browse_test_folder(self):
        dir = self.open_dir()
        if not dir:
            return
        self.json_testdataset_edit.setText(dir.as_posix())

    def browse_yolo_folder(self):
        dir = self.open_dir()
        if not dir:
            return
        self.json_yolodataset_edit.setText(dir.as_posix())

    def browse_annotation_folder(self):
        dir = self.open_dir()
        if not dir:
            return
        self.json_annotationdataset_edit.setText(dir.as_posix())

    def browse_detector_model(self):
        filename = self.open_file_any()
        if not filename:
            return
        self.json_detectormodel_edit.setText(filename)



    def open_file_json(self):
        filename = None
        dialog = QFileDialog()
        #dialog.setDirectory(r'C:\images')
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("Json (*.json )")
        dialog.setViewMode(QFileDialog.ViewMode.List)
        
        filename, ok = dialog.getOpenFileName()
        if filename:
            self.json_filename_edit.setText(filename)
            self.load_json_file(filename)
        #     filename = filenames[0] #self.file_list.addItems([str(Path(filename)) for filename in filenames])


        return filename
    
    def load_json_file(self, filename):
        
        self.jsonLoaded.emit(filename)

    def update_json_values(self):
        self.json_modelname_edit.setText(self.parent_wnd.model_json.model.model_name)
        self.json_filename_edit.setText(self.parent_wnd.model_json.model.model_filename)
        self.json_encoderfile_edit.setText(self.parent_wnd.model_json.model.encoder_filename)
        self.json_traindataset_edit.setText(self.parent_wnd.model_json.model.model_train_dataset)
        self.json_testdataset_edit.setText(self.parent_wnd.model_json.model.model_test_dataset)
        self.json_annotationdataset_edit.setText(self.parent_wnd.model_json.model.annotation_dataset_path)
        self.json_detectormodel_edit.setText(self.parent_wnd.model_json.model.detector_model_path)
        self.json_yolodataset_edit.setText(self.parent_wnd.model_json.model.yolo_dataset_path)
        # Join the list of classes into a string for display
        self.json_classes_edit.setText("".join(self.parent_wnd.model_json.model.model_classes))
        self.json_epochs_edit.setValue(int(self.parent_wnd.model_json.model.train_epochs))
        self.json_imageheight_edit.setValue(int(self.parent_wnd.model_json.model.image_height))
        self.json_imagewidth_edit.setValue(int(self.parent_wnd.model_json.model.image_width))

         
    




        
