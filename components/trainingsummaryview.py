import sys
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QLineEdit,
    QFrame,
    QStyle,
    QComboBox)
from PySide6.QtGui import QFont, QIcon


class TrainingSummaryView(QWidget):
    """
    A widget to display a summary of the model training configuration
    and dataset, with a button to start the training process.
    """

    startTrainingClicked = Signal(str)
    startDetectorTrainingClicked = Signal()
    prepareYoloDataClicked = Signal()
    prepareRecognitionDataClicked = Signal()
    prepareEasyOcrDataClicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # --- Configuration Summary Group ---
        config_group = QGroupBox("Sumário da Configuração")
        config_layout = QFormLayout(config_group)

        self.model_name_label = QLabel("N/A")
        self.epochs_label = QLabel("N/A")
        self.image_size_label = QLabel("N/A")
        self.encoder_path_edit = QLineEdit("N/A")
        self.encoder_path_edit.setFixedWidth(400)
        self.encoder_path_edit.setReadOnly(True)

        self.library_select = QComboBox()
        self.library_select.addItems(["TensorFlow", "PyTorch", "EasyOCR"])

        config_layout.addRow("Nome do Modelo:", self.model_name_label)
        config_layout.addRow("Épocas de Treinamento:", self.epochs_label)
        config_layout.addRow("Dimensões da Imagem:", self.image_size_label)
        config_layout.addRow("Arquivo do Codificador:", self.encoder_path_edit)
        config_layout.addRow("Biblioteca de Treino:", self.library_select)

        # --- Dataset Summary Group ---
        dataset_group = QGroupBox("Sumário do Dataset")
        dataset_layout = QFormLayout(dataset_group)

        self.num_classes_label = QLabel("N/A")
        self.total_images_label = QLabel("N/A")
        self.classes_list_label = QLabel("N/A")
        self.train_path_edit = QLineEdit("N/A")
        self.train_path_edit.setFixedWidth(400)
        self.train_path_edit.setReadOnly(True)
        self.test_path_edit = QLineEdit("N/A")
        self.test_path_edit.setFixedWidth(400)
        self.test_path_edit.setReadOnly(True)

        dataset_layout.addRow("Número de Classes:", self.num_classes_label)
        dataset_layout.addRow("Total de Imagens:", self.total_images_label)
        dataset_layout.addRow("Classes:", self.classes_list_label)
        dataset_layout.addRow("Dataset de Treinamento:", self.train_path_edit)
        dataset_layout.addRow("Dataset de Validação:", self.test_path_edit)

        # --- Actions Group ---
        actions_group = QGroupBox("Ações")
        actions_layout = QHBoxLayout(actions_group)
        actions_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.start_button = QPushButton("Iniciar Treinamento")
        start_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        self.start_button.setIcon(start_icon)
        self.start_button.setMinimumHeight(40)
        font = self.start_button.font()
        font.setPointSize(12)
        font.setBold(True)
        self.start_button.setFont(font)        
        self.start_button.clicked.connect(self.on_start_clicked)
        self.start_button.setEnabled(False)

        self.train_detector_button = QPushButton("Train Detector")
        detector_icon = QIcon(":icons/icons/crosshair.svg")
        self.train_detector_button.setIcon(detector_icon)
        self.train_detector_button.setMinimumHeight(40)
        self.train_detector_button.setFont(font)
        self.train_detector_button.clicked.connect(self.startDetectorTrainingClicked)
        self.train_detector_button.setEnabled(False) # Enable when a model is loaded

        self.prepare_yolo_button = QPushButton("Prepare Detector(YOLO) Data")
        prepare_icon = QIcon(":icons/icons/zap.svg")
        self.prepare_yolo_button.setIcon(prepare_icon)
        self.prepare_yolo_button.setMinimumHeight(40)
        self.prepare_yolo_button.setFont(font)
        self.prepare_yolo_button.clicked.connect(self.prepareYoloDataClicked)
        self.prepare_yolo_button.setEnabled(False)

        self.prepare_rec_button = QPushButton("Prepare Recognition Data")
        rec_icon = QIcon(":icons/icons/image.svg")
        self.prepare_rec_button.setIcon(rec_icon)
        self.prepare_rec_button.setMinimumHeight(40)
        self.prepare_rec_button.setFont(font)
        self.prepare_rec_button.clicked.connect(self.prepareRecognitionDataClicked)
        self.prepare_rec_button.setEnabled(False)
        
        self.prepare_easyocr_button = QPushButton("Prepare EasyOCR Data")
        easyocr_icon = QIcon(":icons/icons/file-text.svg")
        self.prepare_easyocr_button.setIcon(easyocr_icon)
        self.prepare_easyocr_button.setMinimumHeight(40)
        self.prepare_easyocr_button.setFont(font)
        self.prepare_easyocr_button.clicked.connect(self.prepareEasyOcrDataClicked)
        self.prepare_easyocr_button.setEnabled(False)

        actions_layout.addWidget(self.start_button)
        actions_layout.addSpacing(20)
        actions_layout.addWidget(self.train_detector_button)
        actions_layout.addWidget(self.prepare_rec_button)
        actions_layout.addWidget(self.prepare_easyocr_button)
        actions_layout.addWidget(self.prepare_yolo_button)

        horiz_groups = QHBoxLayout()
        horiz_groups.addWidget(config_group)
        horiz_groups.addWidget(dataset_group)

        main_layout.addLayout(horiz_groups)
        main_layout.addStretch()
        main_layout.addWidget(actions_group)

    def update_summary(self, model_data, dataset_summary):
        """
        Updates the labels with the latest configuration and dataset info.
        :param model_data: A ModelAi object from ModelJson.
        :param dataset_summary: A dict like {'classes': int, 'images': int}.
        """
        if model_data:
            self.model_name_label.setText(f"<b>{model_data.model_name}</b>")
            self.epochs_label.setText(f"<b>{model_data.train_epochs}</b>")
            self.image_size_label.setText(f"<b>{model_data.image_width} x {model_data.image_height}</b>")
            self.encoder_path_edit.setText(model_data.encoder_filename)
            self.train_path_edit.setText(model_data.model_train_dataset)
            self.test_path_edit.setText(model_data.model_test_dataset)
            # Join list of classes into a displayable string
            self.classes_list_label.setText(f"<b>{''.join(model_data.model_classes)}</b>")
            self.start_button.setEnabled(True)
            self.prepare_rec_button.setEnabled(True)
            self.prepare_yolo_button.setEnabled(True)
            self.train_detector_button.setEnabled(True)
            self.prepare_easyocr_button.setEnabled(True)
        else:
            self.model_name_label.setText("N/A")
            self.epochs_label.setText("N/A")
            self.image_size_label.setText("N/A")
            self.encoder_path_edit.setText("N/A")
            self.train_path_edit.setText("N/A")
            self.test_path_edit.setText("N/A")
            self.classes_list_label.setText("N/A")
            self.start_button.setEnabled(False)
            self.prepare_rec_button.setEnabled(False)
            self.prepare_yolo_button.setEnabled(False)
            self.train_detector_button.setEnabled(False)
            self.prepare_easyocr_button.setEnabled(False)

        if dataset_summary:
            num_classes = dataset_summary.get('classes', 'N/A')
            num_images = dataset_summary.get('images', 'N/A')
            self.num_classes_label.setText(f"<b>{num_classes}</b>")
            self.total_images_label.setText(f"<b>{num_images}</b>")
        else:
            self.num_classes_label.setText("N/A")
            self.total_images_label.setText("N/A")

    def on_start_clicked(self):
        selected_lib = self.library_select.currentText()
        print(f"DEBUG: TrainingSummaryView emitting startTrainingClicked with '{selected_lib}'")
        self.startTrainingClicked.emit(selected_lib)


if __name__ == '__main__':
    # Example of how to use the widget
    app = QApplication(sys.argv)
    widget = TrainingSummaryView()
    # widget.update_summary(model_data_obj, {'classes': 5, 'images': 123})
    widget.resize(400, 300)
    widget.show()
    sys.exit(app.exec())