import os
import cv2
import numpy as np
import csv

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    nn = None
    Dataset = object

# Define classes conditionally or as None
if torch:
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes, input_h=28, input_w=28):
            super(SimpleCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            
            # Calculate linear input size based on input dimensions
            final_h = max(1, input_h // 8)
            final_w = max(1, input_w // 8)
            linear_input_size = 64 * final_h * final_w

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(linear_input_size, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    class EasyOCRCharNet(nn.Module):
        """
        A custom CNN architecture for EasyOCR character recognition customization.
        """
        def __init__(self, num_classes, input_h, input_w):
            super(EasyOCRCharNet, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            
            # Calculate linear input size
            final_h = input_h // 8
            final_w = input_w // 8
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * final_h * final_w, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    class CharDataset(Dataset):
        def __init__(self, data_list):
            self.data_list = data_list

        def __len__(self):
            return len(self.data_list)

        def __getitem__(self, idx):
            img, label = self.data_list[idx]
            # Convert (H, W, 1) to (1, H, W) for PyTorch
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            return img, label

    class EasyOCRDataset(Dataset):
        def __init__(self, root_dir, width, height, transform=None):
            self.root_dir = root_dir
            self.width = width
            self.height = height
            self.data = []
            self.class_names = set()
            
            csv_path = os.path.join(root_dir, 'labels.csv')
            if os.path.exists(csv_path):
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader, None) # Skip header
                    for row in reader:
                        if len(row) >= 2:
                            self.data.append((row[0], row[1]))
                            self.class_names.add(row[1])
            
            self.class_names = sorted(list(self.class_names))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img_rel_path, label = self.data[idx]
            img_path = os.path.join(self.root_dir, img_rel_path)
            image = cv2.imread(img_path)
            
            if image is None:
                image = np.zeros((self.height, self.width), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Scale image to (self.width, self.height) without cropping
                if image.shape[0] != self.height or image.shape[1] != self.width:
                    image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
            
            image = image / 255.0
            # Add channel dim: (1, H, W) for PyTorch
            image = np.expand_dims(image, axis=0)
            
            return torch.from_numpy(image).float(), self.class_to_idx[label]

    def load_pytorch_model_with_class_mismatch_handling(model, model_path, device, num_classes):
        """
        Load a PyTorch model's state_dict, handling class mismatches gracefully.
        If the saved model has a different number of output classes, loads only the 
        compatible feature layers and reinitializes the classifier.
        
        Args:
            model: The PyTorch model instance to load into
            model_path: Path to the .pth file
            device: Device to use (cpu or cuda)
            num_classes: Expected number of classes for the current model
        
        Returns:
            Tuple (success: bool, message: str, requires_retraining: bool)
        """
        try:
            # Try direct load first
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
            return True, "Model loaded successfully.", False
        except RuntimeError as e:
            error_str = str(e)
            # Check if it's a class mismatch error
            if "size mismatch" in error_str and "classifier" in error_str:
                try:
                    print(f"Detected class mismatch. Attempting to load compatible layers...")
                    checkpoint = torch.load(model_path, map_location=device)
                    
                    # Build a new state dict by excluding classifier mismatches
                    new_state_dict = {}
                    incompatible_keys = []
                    
                    for key, value in checkpoint.items():
                        try:
                            # Try to load this parameter
                            current_param = dict(model.named_parameters())[key]
                            if value.shape == current_param.shape:
                                new_state_dict[key] = value
                            else:
                                incompatible_keys.append(key)
                                print(f"  Skipping {key}: shape {value.shape} != {current_param.shape}")
                        except KeyError:
                            incompatible_keys.append(key)
                    
                    # Load the compatible parameters
                    model.load_state_dict(new_state_dict, strict=False)
                    msg = f"Model loaded with class mismatch detected. {len(incompatible_keys)} layers reinitialized. Retraining recommended."
                    print(msg)
                    return True, msg, True
                except Exception as inner_e:
                    return False, f"Failed to handle class mismatch: {inner_e}", False
            else:
                return False, f"Error loading model: {e}", False
        except Exception as e:
            return False, f"Unexpected error loading model: {e}", False

else:
    SimpleCNN = None
    EasyOCRCharNet = None
    CharDataset = None
    EasyOCRDataset = None