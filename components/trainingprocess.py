import os
import sys
import random
import json
import time
import cv2
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
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    print("TensorFlow/Keras not found. Training will not be available.")
    tf = None

def _calculate_iou(boxA, boxB):
    # From annotation [x, y, w, h] to [x1, y1, x2, y2]
    boxA_coords = [boxA['x'], boxA['y'], boxA['x'] + boxA['w'], boxA['y'] + boxA['h']]
    # YOLO box is already [x1, y1, x2, y2]
    inter_x1 = max(boxA_coords[0], boxB[0])
    inter_y1 = max(boxA_coords[1], boxB[1])
    inter_x2 = min(boxA_coords[2], boxB[2])
    inter_y2 = min(boxA_coords[3], boxB[3])
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    boxA_area = (boxA_coords[2] - boxA_coords[0]) * (boxA_coords[3] - boxA_coords[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = inter_area / float(boxA_area + boxB_area - inter_area) if float(boxA_area + boxB_area - inter_area) > 0 else 0
    return iou

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
        self.log_signal(log_message)

def cnn_data_generator(image_paths, model_data, detector_model):
    """
    A generator that yields preprocessed character images and their labels
    by using the YOLO detector to find characters.
    """
    img_h, img_w = int(model_data.image_height), int(model_data.image_width)
    class_names = sorted(model_data.model_classes)
    char_to_int = {char: i for i, char in enumerate(class_names)}

    for img_path in image_paths:
        json_path = img_path + ".json"
        if not os.path.exists(json_path):
            continue

        with open(json_path, 'r') as f:
            annotations = json.load(f)

        source_image = cv2.imread(img_path)
        if source_image is None:
            continue
        
        gray_img = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

        # Use detector to get bounding boxes
        detection_results = detector_model(source_image, verbose=False)
        boxes = detection_results[0].boxes.xyxy.cpu().numpy()

        # Match detected boxes with annotations using IoU
        annotation_list = list(annotations.values())

        for detected_box in boxes:
            best_match = None
            highest_iou = 0.5  # IoU threshold

            for ann in annotation_list:
                iou = _calculate_iou(ann['box'], detected_box)
                if iou > highest_iou:
                    highest_iou = iou
                    best_match = ann

            if best_match:
                true_char = best_match['char']
                if true_char == '?' or true_char not in char_to_int:
                    continue

                x_min, y_min, x_max, y_max = map(int, detected_box)

                # This is the exact same preprocessing as in InferenceView
                char_img = gray_img[y_min:y_max, x_min:x_max]
                h, w = char_img.shape
                if h > w:
                    pad = (h - w) // 2
                    char_img = cv2.copyMakeBorder(char_img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[0])
                elif w > h:
                    pad = (w - h) // 2
                    char_img = cv2.copyMakeBorder(char_img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0])
                
                resized_char = cv2.resize(char_img, (img_w, img_h), interpolation=cv2.INTER_AREA)
                
                # Normalize and expand dims for the model
                normalized_char = resized_char / 255.0
                input_data = np.expand_dims(normalized_char, axis=-1)
                
                yield input_data, char_to_int[true_char]


def train_model_cnn(model_data, progress_callback, log_callback):
    """
    Trains a real CNN model based on the provided configuration.
    """
    if not tf:
        log_callback("TensorFlow is not installed. Cannot proceed with training.")
        return None

    if not YOLO:
        log_callback("Error: 'ultralytics' package not found for data generation.")
        return None

    epochs = int(model_data.train_epochs)
    img_height = int(model_data.image_height)
    img_width = int(model_data.image_width)
    annotation_dir = model_data.annotation_dataset_path
    batch_size = 32

    log_callback(f"--- Starting Training: {model_data.model_name} ---\n")
    log_callback(f"Using Annotation Dataset: {annotation_dir}")
    log_callback(f"Image Dimensions: {img_width}x{img_height}, Epochs: {epochs}\n")

    # --- 1. Load Detector and Prepare Data Files ---
    try:
        detector_model = YOLO(model_data.detector_model_path)
        log_callback("Detector model loaded successfully for data generation.")

        image_files = [os.path.join(annotation_dir, f) for f in os.listdir(annotation_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(image_files)
        split_index = int(len(image_files) * 0.8)
        train_files = image_files[:split_index]
        val_files = image_files[split_index:]

        log_callback(f"Found {len(image_files)} annotated images. Splitting into {len(train_files)} train and {len(val_files)} validation.")

    except Exception as e:
        log_callback(f"Error preparing data: {e}")
        return None, None

    # --- 2. Create tf.data.Dataset from generator ---
    class_names = sorted(model_data.model_classes)
    num_classes = len(class_names)
    log_callback(f"Using {num_classes} classes: {', '.join(class_names)}\n")

    output_signature = (
        tf.TensorSpec(shape=(img_height, img_width, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )

    train_ds = tf.data.Dataset.from_generator(
        lambda: cnn_data_generator(train_files, model_data, detector_model),
        output_signature=output_signature
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        lambda: cnn_data_generator(val_files, model_data, detector_model),
        output_signature=output_signature
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # --- 3. Build CNN Model ---
    model = keras.Sequential([
        layers.Input(shape=(img_height, img_width, 1)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(int(num_classes), activation='softmax')
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
    
    return final_stats, class_names


class TrainingWorker(QObject):
    """
    A worker that runs the training process in a separate thread.
    """
    progress = Signal(int) # Emits training progress percentage
    log = Signal(str)      # Emits log messages
    finished = Signal(object)  # Emits a tuple (stats, class_names) when done

    def __init__(self, model_data):
        super().__init__()
        self.model_data = model_data

    @Slot()
    def run(self):
        """Starts the training process."""
        try:
            stats, class_names = train_model_cnn(self.model_data, self.progress, self.log)
            self.finished.emit((stats, class_names))
        except Exception as e:
            self.log.emit(f"\n--- An error occurred ---\n{e}")
            self.finished.emit(({}, []))


class TrainingProcessDialog(QDialog):
    """
    A dialog to show the progress of the model training process.
    """
    trainingCompleted = Signal(object) # Signal to emit a tuple (stats, class_names) to parent
    validationTestCompleted = Signal(list) # Signal to emit validation results to parent

    def __init__(self, model_data, parent=None):
        super().__init__(parent)
        self.model_data = model_data
        self.train_files = [] # To store file lists for validation
        self.val_files = []
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
        self.run_test_button.clicked.connect(self.run_validation_test)

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
    def on_training_finished(self, result):
        """Handles the end of the training process."""
        self.thread.quit()
        
        stats, class_names = result

        if stats:
            # Pass both stats and class_names to the parent
            self.trainingCompleted.emit((stats, class_names))
            self.log_area.append("\n--- Training Statistics ---")
            for key, value in stats.items():
                self.log_area.append(f"{key}: {value:.4f}")
            self.run_test_button.setEnabled(True)
        else:
            # Emit empty results on failure
            self.trainingCompleted.emit(({}, []))
            self.log_area.append("\nTraining failed. See logs for details.")

    def run_validation_test(self):
        """Loads the trained model and runs predictions on a subset of validation data."""
        self.log_area.append("\n--- Running Validation Test ---")
        self.run_test_button.setEnabled(False)

        if not tf:
            self.log_area.append("TensorFlow is not installed. Cannot run validation.")
            return

        try:
            # Load the trained model
            model = keras.models.load_model(self.model_data.get_model_save_path())
            # Get the class names directly from the model data that was updated after training
            class_names = self.model_data.model_classes

            # --- This part was broken, let's fix it ---
            # We need access to the validation file list and the detector model.
            # We can re-create them here for the validation test.
            detector_model = YOLO(self.model_data.detector_model_path)
            annotation_dir = self.model_data.annotation_dataset_path
            image_files = [os.path.join(annotation_dir, f) for f in os.listdir(annotation_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            random.seed(42) # Use the same seed to get the same split
            random.shuffle(image_files)
            split_index = int(len(image_files) * 0.8)
            val_files = image_files[split_index:]

            # Create a validation dataset
            val_ds = tf.data.Dataset.from_generator(
                lambda: cnn_data_generator(val_files, self.model_data, detector_model),
                output_signature=(tf.TensorSpec(shape=(self.model_data.image_height, self.model_data.image_width, 1), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.int32))
            ).batch(1) # Predict one by one

            results = []
            # Take a small sample for display
            for i, (images, labels) in enumerate(val_ds.take(min(20, len(val_files)))):
                img_array = (images.numpy()[0] * 255).astype("uint8")
                true_label_idx = labels.numpy()[0]
                true_label = class_names[true_label_idx]

                prediction = model.predict(images, verbose=0)
                predicted_label_idx = np.argmax(prediction[0])
                predicted_label = class_names[predicted_label_idx]
                confidence = np.max(prediction[0]) * 100

                results.append({
                    'image_array': img_array,
                    'actual_label': true_label,
                    'predicted_label': predicted_label,
                    'confidence': confidence
                })
            
            self.log_area.append(f"Generated {len(results)} validation samples.")
            self.validationTestCompleted.emit(results)

        except Exception as e:
            self.log_area.append(f"Error during validation: {e}")
            QMessageBox.critical(self, "Validation Error", f"An error occurred during validation:\n{e}")
        finally:
            self.run_test_button.setEnabled(True)

    def closeEvent(self, event):
        """Ensure the thread is stopped if the dialog is closed prematurely."""
        if self.thread.isRunning():
            self.worker.disconnect() # Disconnect signals to avoid race conditions on close
            self.thread.quit() # Ask the event loop to stop
            self.thread.wait(500) # Wait a bit for it to finish
        event.accept()
        num_classes = len(class_names)
        log_callback(f"Found {num_classes} classes: {', '.join(class_names)}\n")
    

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
            layers.Dense(int(num_classes), activation='softmax')
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
        
        # Return both stats and the discovered class names
        # The class names are sorted alphabetically by image_dataset_from_directory
        return final_stats, class_names


class TrainingWorker(QObject):
    """
    A worker that runs the training process in a separate thread.
    """
    progress = Signal(int) # Emits training progress percentage
    log = Signal(str)      # Emits log messages
    finished = Signal(object)  # Emits a tuple (stats, class_names) when done

    def __init__(self, model_data):
        super().__init__()
        self.model_data = model_data

    @Slot()
    def run(self):
        """Starts the training process."""
        try:
            stats, class_names = train_model_cnn(self.model_data, self.progress, self.log.emit)
            self.finished.emit((stats, class_names))
        except Exception as e:
            self.log.emit(f"\n--- An error occurred ---\n{e}")
            self.finished.emit(({}, []))


class TrainingProcessDialog(QDialog):
    """
    A dialog to show the progress of the model training process.
    """
    trainingCompleted = Signal(object) # Signal to emit a tuple (stats, class_names) to parent
    validationTestCompleted = Signal(list) # Signal to emit validation results to parent

    def __init__(self, model_data, parent=None):
        super().__init__(parent)
        self.model_data = model_data
        self.train_files = [] # To store file lists for validation
        self.val_files = []
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
        self.run_test_button.clicked.connect(self.run_validation_test)

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
    def on_training_finished(self, result):
        """Handles the end of the training process."""
        self.thread.quit()
        
        stats, class_names = result

        if stats:
            # Pass both stats and class_names to the parent
            self.trainingCompleted.emit((stats, class_names))
            self.log_area.append("\n--- Training Statistics ---")
            for key, value in stats.items():
                self.log_area.append(f"{key}: {value:.4f}")
            self.run_test_button.setEnabled(True)
        else:
            # Emit empty results on failure
            self.trainingCompleted.emit(({}, []))
            self.log_area.append("\nTraining failed. See logs for details.")

    def run_validation_test(self):
        """Loads the trained model and runs predictions on a subset of validation data."""
        self.log_area.append("\n--- Running Validation Test ---")
        self.run_test_button.setEnabled(False)

        if not tf:
            self.log_area.append("TensorFlow is not installed. Cannot run validation.")
            return

        try:
            # Load the trained model
            model = keras.models.load_model(self.model_data.get_model_save_path())
            # Get the class names directly from the model data that was updated after training
            class_names = self.model_data.model_classes

            # Re-create the detector model for validation
            detector_model = YOLO(self.model_data.detector_model_path)

            # Create a validation dataset
            val_ds = tf.data.Dataset.from_generator(
                lambda: cnn_data_generator(self.val_files, self.model_data, detector_model),
                output_signature=(tf.TensorSpec(shape=(self.model_data.image_height, self.model_data.image_width, 1), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.int32))
            ).batch(1) # Predict one by one

            results = []
            # Take a small sample for display
            for i, (images, labels) in enumerate(val_ds.take(min(20, len(self.val_files)))):
                img_array = (images.numpy()[0] * 255).astype("uint8")
                true_label_idx = labels.numpy()[0]
                true_label = class_names[true_label_idx]

                prediction = model.predict(images, verbose=0)
                predicted_label_idx = np.argmax(prediction[0])
                predicted_label = class_names[predicted_label_idx]
                confidence = np.max(prediction[0]) * 100

                results.append({
                    'image_array': img_array,
                    'actual_label': true_label,
                    'predicted_label': predicted_label,
                    'confidence': confidence
                })
            
            self.log_area.append(f"Generated {len(results)} validation samples.")
            self.validationTestCompleted.emit(results)

        except Exception as e:
            self.log_area.append(f"Error during validation: {e}")
            QMessageBox.critical(self, "Validation Error", f"An error occurred during validation:\n{e}")
        finally:
            self.run_test_button.setEnabled(True)

    def closeEvent(self, event):
        """Ensure the thread is stopped if the dialog is closed prematurely."""
        if self.thread.isRunning():
            self.worker.disconnect() # Disconnect signals to avoid race conditions on close
            self.thread.quit() # Ask the event loop to stop
            self.thread.wait(500) # Wait a bit for it to finish
        event.accept()
