import sys
import time
from PySide6.QtCore import QObject, Signal, Slot, QThread, Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QProgressBar,
    QTextEdit,
    QPushButton,
    QDialogButtonBox,
    QMessageBox,
    QHBoxLayout,
)

# This is a placeholder for your actual training logic.
# You would replace the content of this function with your TensorFlow/Keras/PyTorch code.
def train_model_placeholder(model_data, progress_callback, log_callback):
    """Placeholder function to simulate model training."""
    epochs = model_data.train_epochs
    log_callback(f"--- Starting Training: {model_data.model_name} ---\n")
    log_callback(f"Dataset: {model_data.model_train_dataset}")
    log_callback(f"Epochs: {epochs}\n")
    time.sleep(1)

    for epoch in range(1, epochs + 1):
        # Simulate training for one epoch
        time.sleep(0.5)
        accuracy = 0.6 + (epoch / epochs) * 0.35 + (time.time() % 0.05)
        loss = 1.0 - (epoch / epochs) * 0.8 + (time.time() % 0.05)

        log_callback(f"Epoch {epoch}/{epochs} - loss: {loss:.4f} - accuracy: {accuracy:.4f}")
        progress_callback.emit(int((epoch / epochs) * 100))

    log_callback("\n--- Training Finished ---")
    final_stats = {"final_accuracy": accuracy, "final_loss": loss}
    return final_stats


class TrainingWorker(QObject):
    """
    A worker that runs the training process in a separate thread.
    """
    progress = Signal(int)
    log = Signal(str)
    finished = Signal(dict)

    def __init__(self, model_data):
        super().__init__()
        self.model_data = model_data

    @Slot()
    def run(self):
        """Starts the training process."""
        try:
            stats = train_model_placeholder(self.model_data, self.progress, self.log.emit)
            self.finished.emit(stats)
        except Exception as e:
            self.log.emit(f"\n--- An error occurred ---\n{e}")
            self.finished.emit({})


class TrainingProcessDialog(QDialog):
    """
    A dialog to show the progress of the model training process.
    """
    def __init__(self, model_data, parent=None):
        super().__init__(parent)
        self.model_data = model_data
        self.setWindowTitle("Training Model")
        self.setMinimumSize(500, 400)
        self.setModal(True)

        # --- Widgets ---
        self.progress_bar = QProgressBar()
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)

        self.run_test_button = QPushButton("Executar Teste de Validação")
        self.run_test_button.setEnabled(False)

        self.close_button = QPushButton("Fechar")
        self.close_button.clicked.connect(self.accept)

        # --- Layout ---
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.run_test_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.log_area)
        layout.addLayout(button_layout)

        # --- Threading Setup ---
        self.setup_training_thread()

    def setup_training_thread(self):
        """Sets up and starts the training worker thread."""
        self.thread = QThread()
        self.worker = TrainingWorker(self.model_data)
        self.worker.moveToThread(self.thread)

        # --- Connections ---
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_training_finished)
        self.worker.progress.connect(self.set_progress)
        self.worker.log.connect(self.append_log)

        # Ensure thread and worker are deleted safely after finishing
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)


        self.thread.start()
        self.log_area.append("Initializing training thread...")

    @Slot(int)
    def set_progress(self, value):
        self.progress_bar.setValue(value)

    @Slot(str)
    def append_log(self, message):
        self.log_area.append(message)

    @Slot(dict)
    def on_training_finished(self, stats):
        """Handles the end of the training process."""
        self.thread.quit()

        if stats:
            self.log_area.append("\n--- Training Statistics ---")
            for key, value in stats.items():
                self.log_area.append(f"{key}: {value:.4f}")
            self.run_test_button.setEnabled(True)
        else:
            self.log_area.append("\nTraining failed. See logs for details.")

        self.run_test_button.clicked.connect(self.run_validation_test)

    def run_validation_test(self):
        """Placeholder for running the test on the validation set."""
        self.log_area.append("\n--- Running Validation Test ---")
        # In a real application, you would likely start another worker thread here.
        # For this example, we'll just simulate it.
        time.sleep(1)
        self.log_area.append("Test Accuracy: 92.5%")
        self.log_area.append("Test Loss: 0.2510")
        self.log_area.append("\n--- Validation Finished ---")
        self.run_test_button.setEnabled(False)

    def closeEvent(self, event):
        """Ensure the thread is stopped if the dialog is closed prematurely."""
        if self.thread.isRunning():
            self.worker.disconnect() # Disconnect signals to avoid race conditions on close
            self.thread.quit() # Ask the event loop to stop
            self.thread.wait(500) # Wait a bit for it to finish
        event.accept()


if __name__ == '__main__':
    # Example of how to use the dialog
    from components.modeljson import ModelAi
    app = QApplication(sys.argv)

    # Create dummy model data
    dummy_model_data = ModelAi(
        model_name="TestCNN",
        model_filename="test.h5",
        encoder_filename="encoder.npy",
        model_train_dataset="/path/to/train",
        model_test_dataset="/path/to/test",
        model_classes="ABC123",
        train_epochs=20,
        image_height=28,
        image_width=28
    )

    dialog = TrainingProcessDialog(dummy_model_data)
    dialog.exec()
    sys.exit(0)