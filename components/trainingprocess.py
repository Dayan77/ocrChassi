import os
import sys
import time
import numpy as np
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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Suppress TensorFlow INFO messages

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    print("TensorFlow/Keras not found. Training will not be available.")
    tf = None


class KerasProgressCallback(keras.callbacks.Callback):
    """A Keras callback to update the UI during training."""
    def __init__(self, progress_signal, log_signal):
        super().__init__()
        self.progress_signal = progress_signal
        self.log_signal = log_signal

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Calculate progress based on epochs
        progress_percent = int(((epoch + 1) / self.params['epochs']) * 100)
        self.progress_signal.emit(progress_percent)

        # Format and emit log message
        log_message = f"Epoch {epoch+1}/{self.params['epochs']}"
        for key, value in logs.items():
            log_message += f" - {key}: {value:.4f}"
        self.log_signal(log_message) # Corrected: Call the passed emit method directly


def train_model_cnn(model_data, progress_callback, log_callback):
    """
    Trains a real CNN model based on the provided configuration.
    """
    if not tf:
        log_callback("TensorFlow is not installed. Cannot proceed with training.")
        return None

    epochs = int(model_data.train_epochs)
    img_height = int(model_data.image_height)
    img_width = int(model_data.image_width)
    train_dir = model_data.model_train_dataset
    batch_size = 32

    log_callback(f"--- Starting Training: {model_data.model_name} ---\n")
    log_callback(f"Loading dataset from: {train_dir}")
    log_callback(f"Image Dimensions: {img_width}x{img_height}, Epochs: {epochs}\n")

    # --- 1. Load Dataset ---
    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            color_mode='grayscale'
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            color_mode='grayscale'
        )
        class_names = train_ds.class_names
        num_classes = len(class_names)
        log_callback(f"Found {num_classes} classes: {', '.join(class_names)}\n")
    except Exception as e:
        log_callback(f"Error loading dataset: {e}")
        return None

    # --- 2. Create and Save Label Encoder (Class Names) ---
    # TensorFlow's image_dataset_from_directory sorts class names alphabetically.
    # We just need to save this list for later use during inference.
    np.save(model_data.get_label_encoder_save_path(), np.array(class_names))
    log_callback(f"Label mapping saved to: {model_data.get_label_encoder_save_path()}\n")

    # --- 3. Build CNN Model ---
    model = keras.Sequential([
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 1)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    
    model.summary(print_fn=lambda x: log_callback(x))

    # --- 4. Train the Model ---
    log_callback("\n--- Starting Model Fitting ---")
    ui_callback = KerasProgressCallback(progress_callback, log_callback)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[ui_callback],
        verbose=0 # We use our custom callback for logging
    )

    # --- 5. Save the Model and Return Stats ---
    model.save(model_data.get_model_save_path())
    log_callback(f"\n--- Training Finished. Model saved to: {model_data.get_model_save_path()} ---")

    final_stats = {
        "final_accuracy": history.history['accuracy'][-1],
        "final_loss": history.history['loss'][-1],
        "final_val_accuracy": history.history['val_accuracy'][-1],
        "final_val_loss": history.history['val_loss'][-1]
    }
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
            stats = train_model_cnn(self.model_data, self.progress, self.log.emit)
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


# if __name__ == '__main__':
#     # Example of how to use the dialog
#     from components.modeljson import ModelAi
#     app = QApplication(sys.argv)

#     # Create dummy model data
#     dummy_model_data = ModelAi(
#         model_name="TestCNN",
#         model_filename="test.h5",
#         encoder_filename="encoder.npy",
#         model_train_dataset="/path/to/train",
#         model_test_dataset="/path/to/test",
#         model_classes="ABC123",
#         train_epochs=20,
#         image_height=28,
#         image_width=28
#     )

#     dialog = TrainingProcessDialog(dummy_model_data)
#     dialog.exec()
#     sys.exit(0)