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

    startTrainingClicked = Signal(str, str)  # (library, architecture)
    startDetectorTrainingClicked = Signal()
    prepareYoloDataClicked = Signal()
    prepareRecognitionDataClicked = Signal()
    prepareEasyOcrDataClicked = Signal()
    reviewRecognitionDataClicked = Signal()
    reviewYoloDataClicked = Signal()
    activeLearningClicked = Signal()
    dataAugmentationClicked = Signal()

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

        self.architecture_select = QComboBox()
        # Display label -> internal key passed to the trainer
        self.architecture_select.addItem(
            "SimpleCNN — rápida, sem regularização (~2,1M params)", "SimpleCNN"
        )
        self.architecture_select.addItem(
            "CNN robusta — BatchNorm + Dropout 0.5 (~4M params)", "EasyOCRCharNet"
        )
        self.architecture_select.setToolTip(
            "SimpleCNN: 3 blocos conv simples (mais rápida, mais sensível a overfit).\n"
            "CNN robusta: mesma profundidade com BatchNorm em cada conv e Dropout no "
            "classificador — costuma generalizar melhor com poucas amostras."
        )

        config_layout.addRow("Nome do Modelo:", self.model_name_label)
        config_layout.addRow("Épocas de Treinamento:", self.epochs_label)
        config_layout.addRow("Dimensões da Imagem:", self.image_size_label)
        config_layout.addRow("Arquitetura PyTorch:", self.architecture_select)
        config_layout.addRow("Arquivo do Codificador:", self.encoder_path_edit)

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

        self.review_rec_button = QPushButton("Revisar Dados Reconhecimento")
        review_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView)
        self.review_rec_button.setIcon(review_icon)
        self.review_rec_button.setMinimumHeight(40)
        self.review_rec_button.setFont(font)
        self.review_rec_button.setToolTip(
            "Abre os crops gerados pela 'Prepare Recognition Data' para validar/mover/deletar."
        )
        self.review_rec_button.clicked.connect(self.reviewRecognitionDataClicked)
        self.review_rec_button.setEnabled(False)

        self.review_yolo_button = QPushButton("Revisar Dados YOLO")
        self.review_yolo_button.setIcon(review_icon)
        self.review_yolo_button.setMinimumHeight(40)
        self.review_yolo_button.setFont(font)
        self.review_yolo_button.setToolTip(
            "Abre as imagens preparadas para o YOLO em train/val para validar/mover/deletar."
        )
        self.review_yolo_button.clicked.connect(self.reviewYoloDataClicked)
        self.review_yolo_button.setEnabled(False)

        self.active_learning_button = QPushButton("Active Learning")
        al_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DialogHelpButton)
        self.active_learning_button.setIcon(al_icon)
        self.active_learning_button.setMinimumHeight(40)
        self.active_learning_button.setFont(font)
        self.active_learning_button.setToolTip(
            "Roda o modelo nas imagens anotadas e lista os casos onde a predição "
            "diverge da anotação ou tem baixa confiança — para você ampliar o "
            "treino exatamente onde está errando."
        )
        self.active_learning_button.clicked.connect(self.activeLearningClicked)
        self.active_learning_button.setEnabled(False)

        self.data_aug_button = QPushButton("Data Augmentation")
        aug_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView)
        self.data_aug_button.setIcon(aug_icon)
        self.data_aug_button.setMinimumHeight(40)
        self.data_aug_button.setFont(font)
        self.data_aug_button.setToolTip(
            "Gera variantes augmentadas (rotação/brilho/ruído) dos crops em "
            "model_train_dataset/<classe>/ e salva no disco. Útil para "
            "expandir manualmente o dataset antes do treino."
        )
        self.data_aug_button.clicked.connect(self.dataAugmentationClicked)
        self.data_aug_button.setEnabled(False)

        actions_layout.addWidget(self.start_button)
        actions_layout.addSpacing(20)
        actions_layout.addWidget(self.train_detector_button)
        actions_layout.addWidget(self.prepare_rec_button)
        actions_layout.addWidget(self.review_rec_button)
        actions_layout.addWidget(self.prepare_easyocr_button)
        actions_layout.addWidget(self.prepare_yolo_button)
        actions_layout.addWidget(self.review_yolo_button)
        actions_layout.addWidget(self.active_learning_button)
        actions_layout.addWidget(self.data_aug_button)

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

            # Reflect the persisted architecture if the model was trained
            # before — falls back to SimpleCNN for legacy models without the
            # field.
            arch = getattr(model_data, "architecture", "SimpleCNN") or "SimpleCNN"
            idx = self.architecture_select.findData(arch)
            if idx >= 0:
                self.architecture_select.setCurrentIndex(idx)
            self.start_button.setEnabled(True)
            self.prepare_rec_button.setEnabled(True)
            self.prepare_yolo_button.setEnabled(True)
            self.train_detector_button.setEnabled(True)
            self.prepare_easyocr_button.setEnabled(True)
            self.review_rec_button.setEnabled(True)
            self.review_yolo_button.setEnabled(True)
            self.active_learning_button.setEnabled(True)
            self.data_aug_button.setEnabled(True)
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
            self.review_rec_button.setEnabled(False)
            self.review_yolo_button.setEnabled(False)
            self.active_learning_button.setEnabled(False)
            self.data_aug_button.setEnabled(False)

        if dataset_summary:
            num_classes = dataset_summary.get('classes', 'N/A')
            num_images = dataset_summary.get('images', 'N/A')
            self.num_classes_label.setText(f"<b>{num_classes}</b>")
            self.total_images_label.setText(f"<b>{num_images}</b>")
        else:
            self.num_classes_label.setText("N/A")
            self.total_images_label.setText("N/A")

    def on_start_clicked(self):
        selected_lib = "PyTorch"
        architecture = self.architecture_select.currentData() or "SimpleCNN"
        print(
            f"DEBUG: TrainingSummaryView emitting startTrainingClicked "
            f"with library='{selected_lib}' architecture='{architecture}'"
        )
        self.startTrainingClicked.emit(selected_lib, architecture)


if __name__ == '__main__':
    # Example of how to use the widget
    app = QApplication(sys.argv)
    widget = TrainingSummaryView()
    # widget.update_summary(model_data_obj, {'classes': 5, 'images': 123})
    widget.resize(400, 300)
    widget.show()
    sys.exit(app.exec())