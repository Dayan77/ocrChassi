import os
import sys
import numpy as np
from PySide6.QtCore import Qt, QSize, Signal, Slot
from PySide6.QtGui import QPixmap, QImage, QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QGroupBox, QListWidget, QListWidgetItem, QApplication,
    QScrollArea, QSizePolicy
)

class ResultImageItem(QWidget):
    """Custom widget for displaying a validation image with its results."""
    def __init__(self, image_array, actual_label, predicted_label, confidence, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # Image Label
        self.image_label = QLabel()
        if image_array is not None:
            # Convert numpy array to QImage
            height, width = image_array.shape[:2]
            if image_array.ndim == 3: # Color image
                bytes_per_line = 3 * width
                q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
            else: # Grayscale image
                q_image = QImage(image_array.data, width, height, width, QImage.Format.Format_Grayscale8)
            
            pixmap = QPixmap.fromImage(q_image)
            # Scale pixmap to fit within a reasonable size, e.g., 64x64
            scaled_pixmap = pixmap.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
        else:
            self.image_label.setText("No Image")
            self.image_label.setFixedSize(64, 64)
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.image_label.setStyleSheet("border: 1px solid gray;")

        # Info Layout
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(2)

        actual_label_text = QLabel(f"Actual: <b>{actual_label}</b>")
        predicted_label_text = QLabel(f"Predicted: <b>{predicted_label}</b>")
        confidence_text = QLabel(f"Confidence: {confidence:.2f}%")

        # Highlight if prediction is incorrect
        if actual_label != predicted_label:
            predicted_label_text.setStyleSheet("color: red; font-weight: bold;")
        else:
            predicted_label_text.setStyleSheet("color: green; font-weight: bold;")

        info_layout.addWidget(actual_label_text)
        info_layout.addWidget(predicted_label_text)
        info_layout.addWidget(confidence_text)

        layout.addWidget(self.image_label)
        layout.addLayout(info_layout)
        layout.addStretch()

        self.setLayout(layout)

class ResultsView(QWidget):
    """
    A widget to display training results summary and validation test results.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # --- Training Summary Group ---
        self.training_summary_group = QGroupBox("Training Summary")
        self.training_summary_layout = QFormLayout(self.training_summary_group)

        self.accuracy_label = QLabel("N/A")
        self.loss_label = QLabel("N/A")
        self.val_accuracy_label = QLabel("N/A")
        self.val_loss_label = QLabel("N/A")

        self.training_summary_layout.addRow("Accuracy:", self.accuracy_label)
        self.training_summary_layout.addRow("Loss:", self.loss_label)
        self.training_summary_layout.addRow("Validation Accuracy:", self.val_accuracy_label)
        self.training_summary_layout.addRow("Validation Loss:", self.val_loss_label)

        # --- Validation Test Results Group ---
        self.validation_results_group = QGroupBox("Validation Test Results")
        self.validation_results_layout = QVBoxLayout(self.validation_results_group)

        self.validation_list_widget = QListWidget()
        self.validation_list_widget.setFlow(QListWidget.Flow.LeftToRight)
        self.validation_list_widget.setWrapping(True)
        self.validation_list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.validation_list_widget.setUniformItemSizes(True)
        self.validation_list_widget.setGridSize(QSize(200, 80)) # Adjust item size
        self.validation_results_layout.addWidget(self.validation_list_widget)

        main_layout.addWidget(self.training_summary_group)
        main_layout.addWidget(self.validation_results_group)
        main_layout.addStretch()

    @Slot(dict)
    def update_training_summary(self, stats):
        """Updates the training summary labels."""
        if stats:
            self.accuracy_label.setText(f"<b>{stats.get('final_accuracy', 'N/A'):.4f}</b>")
            self.loss_label.setText(f"<b>{stats.get('final_loss', 'N/A'):.4f}</b>")
            self.val_accuracy_label.setText(f"<b>{stats.get('final_val_accuracy', 'N/A'):.4f}</b>")
            self.val_loss_label.setText(f"<b>{stats.get('final_val_loss', 'N/A'):.4f}</b>")
        else:
            self.accuracy_label.setText("N/A")
            self.loss_label.setText("N/A")
            self.val_accuracy_label.setText("N/A")
            self.val_loss_label.setText("N/A")

    @Slot(list)
    def update_validation_results(self, results):
        """
        Updates the list widget with validation test results.
        Results is a list of dicts: [{'image_path', 'actual_label', 'predicted_label', 'confidence'}]
        """
        self.validation_list_widget.clear()
        if results:
            for item_data in results:
                item = QListWidgetItem(self.validation_list_widget)
                custom_widget = ResultImageItem(
                    item_data['image_array'],
                    item_data['actual_label'],
                    item_data['predicted_label'],
                    item_data['confidence']
                )
                item.setSizeHint(custom_widget.sizeHint())
                self.validation_list_widget.addItem(item)
                self.validation_list_widget.setItemWidget(item, custom_widget)
        else:
            self.validation_list_widget.addItem(QListWidgetItem("No validation results to display."))