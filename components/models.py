import os
import cv2
import numpy as np
import csv
import random

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    nn = None
    Dataset = object


IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')


def list_class_folders(root_dir):
    """Return sorted list of class names (subdirectories) inside root_dir."""
    if not root_dir or not os.path.isdir(root_dir):
        return []
    return sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )


def count_class_samples(root_dir):
    """Return {class_name: image_count} for a recog-style folder layout."""
    counts = {}
    for class_name in list_class_folders(root_dir):
        class_path = os.path.join(root_dir, class_name)
        n = sum(1 for f in os.listdir(class_path) if f.lower().endswith(IMAGE_EXTS))
        counts[class_name] = n
    return counts


def expand_bbox(x1, y1, x2, y2, img_h, img_w, pad_pct):
    """Expand a bbox outward by `pad_pct` percent of its own width/height,
    then clamp to the image bounds. Returns ints.

    Used by both the inference path and the recognition-data preparation
    path so train and inference see crops with the same padding policy.
    `pad_pct` is a percentage (e.g. 10 means each side grows by 10% of the
    box dimension on that axis).
    """
    if pad_pct <= 0:
        return int(max(0, x1)), int(max(0, y1)), int(min(img_w, x2)), int(min(img_h, y2))
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    dx = bw * (pad_pct / 100.0)
    dy = bh * (pad_pct / 100.0)
    nx1 = int(max(0, x1 - dx))
    ny1 = int(max(0, y1 - dy))
    nx2 = int(min(img_w, x2 + dx))
    ny2 = int(min(img_h, y2 + dy))
    return nx1, ny1, nx2, ny2


def filter_detections(boxes, confs,
                      iou_thresh=0.4, iomin_thresh=0.5,
                      height_low=0.6, height_high=1.6,
                      vertical_dev_factor=0.6,
                      max_gap_factor=2.5):
    """Filter raw detector boxes to keep a single horizontal chain of
    characters. Returns numpy array of [x1, y1, x2, y2] sorted left-to-right.

    Pipeline:
      1. Custom NMS (IoU + Intersection-over-Min).
      2. Height consistency around median.
      3. Vertical alignment around median y-center.
      4. Horizontal chain pruning: iteratively drop isolated boxes.

    Pure function so both InferenceView (aba Modelos) and the production
    InferenceWorker (tela Produção) reuse the exact same filter.
    """
    if boxes is None or len(boxes) == 0:
        return np.array([])

    candidates = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        candidates.append({
            'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2),
            'w': float(x2 - x1), 'h': float(y2 - y1),
            'cx': float((x1 + x2) / 2), 'cy': float((y1 + y2) / 2),
            'conf': float(confs[i]) if confs is not None and i < len(confs) else 0.0,
        })

    candidates.sort(key=lambda x: x['conf'], reverse=True)
    keep = []
    for c in candidates:
        discard = False
        for k in keep:
            xA = max(c['x1'], k['x1'])
            yA = max(c['y1'], k['y1'])
            xB = min(c['x2'], k['x2'])
            yB = min(c['y2'], k['y2'])
            inter = max(0.0, xB - xA) * max(0.0, yB - yA)
            if inter > 0:
                area_c = c['w'] * c['h']
                area_k = k['w'] * k['h']
                iou = inter / (area_c + area_k - inter)
                io_min = inter / min(area_c, area_k) if min(area_c, area_k) > 0 else 0
                if iou > iou_thresh or io_min > iomin_thresh:
                    discard = True
                    break
        if not discard:
            keep.append(c)
    candidates = keep
    if not candidates:
        return np.array([])

    median_h = float(np.median([c['h'] for c in candidates]))
    candidates = [c for c in candidates
                  if height_low * median_h < c['h'] < height_high * median_h]
    if not candidates:
        return np.array([])

    median_cy = float(np.median([c['cy'] for c in candidates]))
    candidates = [c for c in candidates
                  if abs(c['cy'] - median_cy) < (median_h * vertical_dev_factor)]
    if not candidates:
        return np.array([])

    while len(candidates) >= 3:
        candidates.sort(key=lambda x: x['cx'])
        cxs = [c['cx'] for c in candidates]
        gaps = [cxs[i + 1] - cxs[i] for i in range(len(cxs) - 1)]
        median_gap = float(np.median(gaps)) if gaps else 0.0
        if median_gap <= 0:
            break
        threshold = max_gap_factor * median_gap
        INF = float('inf')
        worst_idx = -1
        worst_gap = 0.0
        for i in range(len(candidates)):
            left = INF if i == 0 else cxs[i] - cxs[i - 1]
            right = INF if i == len(candidates) - 1 else cxs[i + 1] - cxs[i]
            nearest = min(left, right)
            if nearest > threshold and nearest > worst_gap:
                worst_gap = nearest
                worst_idx = i
        if worst_idx < 0:
            break
        candidates.pop(worst_idx)

    candidates.sort(key=lambda x: x['x1'])
    return np.array([[c['x1'], c['y1'], c['x2'], c['y2']] for c in candidates])


def preprocess_char_crop(gray_crop, target_h, target_w, apply_clahe=True,
                         clahe_clip=2.0, clahe_grid=(8, 8)):
    """Single source of truth for turning a raw grayscale character crop into
    the tensor-ready uint8 image. Used by both the recognition-data prep
    (so the on-disk train set looks like this) and the inference path
    (so the CNN sees the same distribution). Order:
      1. pad-to-square with zeros (preserves aspect ratio).
      2. resize to (target_w, target_h) with INTER_AREA.
      3. optional CLAHE for local contrast.
    Returns uint8 HxW; the caller normalizes to [0,1] if needed.
    """
    if gray_crop is None or gray_crop.size == 0:
        return None
    img = gray_crop
    h, w = img.shape[:2]
    if h > w:
        pad = (h - w) // 2
        img = cv2.copyMakeBorder(img, 0, 0, pad, h - w - pad, cv2.BORDER_CONSTANT, value=0)
    elif w > h:
        pad = (w - h) // 2
        img = cv2.copyMakeBorder(img, pad, w - h - pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
    if img.shape[0] != target_h or img.shape[1] != target_w:
        img = cv2.resize(img, (int(target_w), int(target_h)), interpolation=cv2.INTER_AREA)
    if apply_clahe:
        try:
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
            img = clahe.apply(img)
        except Exception:
            pass
    return img


def _augment_char_gray(img, rng):
    """Light augmentation for character crops. Input/output: uint8 grayscale HxW."""
    h, w = img.shape[:2]

    # Small affine: rotation in [-8, 8] deg + scale [0.92, 1.08] + translation up to 5%
    angle = rng.uniform(-8.0, 8.0)
    scale = rng.uniform(0.92, 1.08)
    tx = rng.uniform(-0.05, 0.05) * w
    ty = rng.uniform(-0.05, 0.05) * h
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    img = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Brightness / contrast jitter
    alpha = rng.uniform(0.85, 1.15)
    beta = rng.uniform(-15.0, 15.0)
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # Light gaussian noise (50% chance)
    if rng.random() < 0.5:
        noise = rng.normal(0.0, 5.0, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


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

    class RecognitionDataset(Dataset):
        """
        Loads character crops from a folder-per-class layout
        (e.g. recog/<class_name>/*.png), grayscale, resized to (height, width).
        Optional augmentation is applied only when augment=True.

        Stores `samples` as list of (path, class_idx) and exposes:
          - class_names: sorted class names
          - class_to_idx
          - class_counts: {class_idx: count}
        """

        def __init__(self, root_dir, width, height, class_names=None, augment=False, seed=42):
            self.root_dir = root_dir
            self.width = int(width)
            self.height = int(height)
            self.augment = bool(augment)
            self._rng = random.Random(seed)
            self._np_rng = np.random.default_rng(seed)

            discovered = list_class_folders(root_dir)
            if class_names is None:
                class_names = discovered
            self.class_names = list(class_names)
            self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

            self.samples = []
            for class_name in discovered:
                if class_name not in self.class_to_idx:
                    continue
                class_idx = self.class_to_idx[class_name]
                class_path = os.path.join(root_dir, class_name)
                for fname in sorted(os.listdir(class_path)):
                    if fname.lower().endswith(IMAGE_EXTS):
                        self.samples.append((os.path.join(class_path, fname), class_idx))

            counts = {i: 0 for i in range(len(self.class_names))}
            for _, idx in self.samples:
                counts[idx] += 1
            self.class_counts = counts

        def __len__(self):
            return len(self.samples)

        def class_counts_by_name(self):
            return {self.class_names[i]: c for i, c in self.class_counts.items()}

        def __getitem__(self, idx):
            path, class_idx = self.samples[idx]
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((self.height, self.width), dtype=np.uint8)
            if img.shape[0] != self.height or img.shape[1] != self.width:
                img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)

            if self.augment:
                img = _augment_char_gray(img, self._np_rng)

            arr = img.astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
            return tensor, class_idx

    def compute_class_weights(class_counts, num_classes, smooth=1.0):
        """
        Inverse-frequency class weights, normalized so the mean is 1.
        Empty classes get the weight of the smallest non-zero class so
        the loss does not blow up when no samples exist.
        """
        counts = np.array(
            [class_counts.get(i, 0) for i in range(num_classes)],
            dtype=np.float64,
        )
        if counts.sum() == 0:
            return torch.ones(num_classes, dtype=torch.float32)
        nonzero = counts[counts > 0]
        floor = nonzero.min() if nonzero.size > 0 else 1.0
        counts = np.where(counts > 0, counts, floor)
        weights = 1.0 / (counts + smooth)
        weights = weights / weights.mean()
        return torch.from_numpy(weights.astype(np.float32))

    def make_sample_weights(targets, class_counts, num_classes):
        """Per-sample weights for WeightedRandomSampler (inverse class frequency)."""
        class_w = compute_class_weights(class_counts, num_classes).numpy()
        return np.array([class_w[t] for t in targets], dtype=np.float32)

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
    RecognitionDataset = None
    compute_class_weights = None
    make_sample_weights = None