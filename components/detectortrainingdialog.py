import os
import yaml
import shutil
from PySide6.QtCore import QObject, Signal, Slot, QThread, Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QProgressBar, QTextEdit,
    QPushButton, QMessageBox, QHBoxLayout
)

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class DetectorTrainingWorker(QObject):
    """
    A worker that runs the YOLO detector training process in a separate thread.
    """
    progress = Signal(int)
    log = Signal(str)
    finished = Signal(str)

    def __init__(self, model_data):
        super().__init__()
        self.model_data = model_data

    def on_train_epoch_end(self, trainer):
        """Callback function for Ultralytics training progress."""
        epoch = trainer.epoch
        total_epochs = trainer.epochs
        progress_percent = int(((epoch + 1) / total_epochs) * 100)
        self.progress.emit(progress_percent)

        # Format and emit log message
        results_dict = trainer.validator.metrics.results_dict
        map50 = results_dict.get('metrics/mAP50', 0)
        log_message = f"Epoch {epoch+1}/{total_epochs} - mAP50: {map50:.4f}"
        self.log.emit(log_message)

    @Slot()
    def run(self):
        """Starts the YOLO training process."""
        if not YOLO:
            self.log.emit("Error: 'ultralytics' package not found. Please install it.")
            self.finished.emit(None)
            return

        try:
            # Use the dedicated path for the YOLO dataset
            yolo_dataset_path = os.path.abspath(self.model_data.yolo_dataset_path)
            data_yaml_path = os.path.join(yolo_dataset_path, "data.yaml")

            if not os.path.exists(data_yaml_path):
                self.log.emit(f"Error: data.yaml not found in '{yolo_dataset_path}'. Please prepare the dataset first.")
                self.finished.emit(None)
                return

            self.log.emit("--- Starting YOLOv8 Detector Training ---")

            # Load a pretrained model
            model = YOLO('yolov8n.pt')

            # Add our custom callback
            model.add_callback("on_train_epoch_end", self.on_train_epoch_end)

            # Train the model
            model.train(
                data=data_yaml_path,
                epochs=self.model_data.train_epochs,
                imgsz=640, # Standard size for YOLO, can be adjusted
                project="runs",
                name="detector_training"
            )

            # The best model is saved automatically in runs/detector_training/weights/best.pt
            # For simplicity, we'll just point to this directory.
            save_dir = "runs/detector_training/weights/best.pt"
            self.log.emit(f"\n--- Training Finished. Best model saved to: {save_dir} ---")
            self.finished.emit(save_dir)

        except Exception as e:
            self.log.emit(f"\n--- An error occurred during detector training ---\n{e}")
            self.finished.emit(None)

class DetectorTrainingDialog(QDialog):
    """
    A dialog to show the progress of the detector model training process.
    """
    def __init__(self, model_data, parent=None):
        super().__init__(parent)
        self.model_data = model_data
        self.setWindowTitle("Training Detector Model")
        self.setMinimumSize(500, 400)
        self.setModal(True)

        # --- Widgets ---
        self.progress_bar = QProgressBar()
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)

        # --- Layout ---
        layout = QVBoxLayout(self)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.log_area)
        layout.addWidget(self.close_button, alignment=Qt.AlignmentFlag.AlignRight)

        # --- Threading Setup ---
        self.setup_training_thread()

    def setup_training_thread(self):
        """Sets up and starts the training worker thread."""
        self.thread = QThread()
        self.worker = DetectorTrainingWorker(self.model_data)
        self.worker.moveToThread(self.thread)

        # --- Connections ---
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_training_finished)
        self.worker.progress.connect(self.set_progress)
        self.worker.log.connect(self.append_log)

        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()
        self.log_area.append("Initializing detector training thread...")

    @Slot(int)
    def set_progress(self, value):
        self.progress_bar.setValue(value)

    @Slot(str)
    def append_log(self, message):
        self.log_area.append(message)

    @Slot(str)
    def on_training_finished(self, result_path):
        """Handles the end of the training process."""
        self.thread.quit()
        if result_path and os.path.exists(result_path):
            destination_path = "yolo_detector.pt"
            should_copy = True

            if os.path.exists(destination_path):
                reply = QMessageBox.question(
                    self,
                    "Confirm Replace",
                    f"A model named '{destination_path}' already exists.\n\n"
                    "Do you want to replace it with the newly trained model?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    should_copy = False
                    QMessageBox.information(self, "Model Not Updated", 
                                            f"The existing '{destination_path}' was not replaced.")

            if should_copy:
                try:
                    shutil.copy(result_path, destination_path)
                    QMessageBox.information(self, "Training Complete",
                                            f"Detector training finished.\nModel '{destination_path}' has been updated successfully.")
                except Exception as e:
                    QMessageBox.critical(self, "Copy Error", f"Could not copy the trained model: {e}")
        else:
            QMessageBox.critical(self, "Training Failed", "Detector training failed. Please check the logs.")

    def closeEvent(self, event):
        if self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(500)
        event.accept()