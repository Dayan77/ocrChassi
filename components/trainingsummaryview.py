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
    QStyle)
from PySide6.QtGui import QFont, QIcon


class TrainingSummaryView(QWidget):
    """
    A widget to display a summary of the model training configuration
    and dataset, with a button to start the training process.
    """

    startTrainingClicked = Signal()

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

        config_layout.addRow("Nome do Modelo:", self.model_name_label)
        config_layout.addRow("Épocas de Treinamento:", self.epochs_label)
        config_layout.addRow("Dimensões da Imagem:", self.image_size_label)
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
        self.start_button.clicked.connect(self.startTrainingClicked.emit)
        self.start_button.setEnabled(False)

        actions_layout.addWidget(self.start_button)

        main_layout.addWidget(config_group)
        main_layout.addWidget(dataset_group)
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
            self.classes_list_label.setText(f"<b>{model_data.model_classes}</b>")
            self.start_button.setEnabled(True)
        else:
            self.model_name_label.setText("N/A")
            self.epochs_label.setText("N/A")
            self.image_size_label.setText("N/A")
            self.encoder_path_edit.setText("N/A")
            self.train_path_edit.setText("N/A")
            self.test_path_edit.setText("N/A")
            self.classes_list_label.setText("N/A")
            self.start_button.setEnabled(False)

        if dataset_summary:
            num_classes = dataset_summary.get('classes', 'N/A')
            num_images = dataset_summary.get('images', 'N/A')
            self.num_classes_label.setText(f"<b>{num_classes}</b>")
            self.total_images_label.setText(f"<b>{num_images}</b>")
        else:
            self.num_classes_label.setText("N/A")
            self.total_images_label.setText("N/A")


if __name__ == '__main__':
    # Example of how to use the widget
    app = QApplication(sys.argv)
    widget = TrainingSummaryView()
    # widget.update_summary(model_data_obj, {'classes': 5, 'images': 123})
    widget.resize(400, 300)
    widget.show()
    sys.exit(app.exec())