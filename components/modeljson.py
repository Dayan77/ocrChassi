
import json
from dataclasses import dataclass, asdict
from typing import List, Optional

import config_ini


@dataclass
class ModelAi():
    model_name: str
    model_filename: str
    encoder_filename: str
    model_train_dataset: str
    model_test_dataset: str
    yolo_dataset_path: str
    annotation_dataset_path: str
    detector_model_path: str
    model_classes: List[str]
    train_epochs: int
    image_height: int
    image_width: int
    
    def get_model_save_path(self) -> str:
        """Returns the full path where the Keras model should be saved."""
        return config_ini.MODEL_SAVE_PATH

    def get_label_encoder_save_path(self) -> str:
        """Returns the full path where the label encoder should be saved."""
        return self.encoder_filename if self.encoder_filename else config_ini.LABEL_ENCODER_SAVE_PATH


    

class ModelJson():
    model = None
    def create_model(self, model_name: str, model_filename: str, encoder_filename: str, model_train_dataset: str, 
                     model_test_dataset: str, yolo_dataset_path: str, annotation_dataset_path: str, detector_model_path: str, model_classes: List[str], train_epochs: int, image_height: int, image_width: int):
        self.model = ModelAi(model_name=model_name, model_filename=model_filename, encoder_filename=encoder_filename, model_train_dataset=model_train_dataset,
                             model_test_dataset=model_test_dataset, yolo_dataset_path=yolo_dataset_path, annotation_dataset_path=annotation_dataset_path, detector_model_path=detector_model_path, model_classes=model_classes, train_epochs=train_epochs, image_height=image_height, image_width=image_width )
        

    def serialize(self):
        model_dict = asdict(self.model)

        json_string = json.dumps(model_dict, indent=4)

        return json_string
    

    def deserialize(self):
        json_string = self.serialize()

        model_parsed = json.loads(json_string)

        return model_parsed
    
    def to_dict(self, obj):
        """Recursively converts a dataclass object to a dictionary."""
        if isinstance(obj, list):
            return [self.to_dict(item) for item in obj]
        if isinstance(obj, dict):
            return {key: self.to_dict(value) for key, value in obj.items()}
        if dataclass.is_dataclass(obj):
            return asdict(obj)
        return obj
    
    def save_to_file(self, filename):
        try:
        
            self.model.model_filename = filename

            model_dict = asdict(self.model)

            with open(self.model.model_filename, 'w') as json_file:
                json.dump(model_dict, json_file, indent=4)

            return True
        
        except FileNotFoundError:
            print(f"Error: The file '{filename}' was not found.")
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from the file '{filename}'. Check for malformed data.")    
        except Exception as e:
            pass
    

    def load_from_file(self, filename):
        try:
            with open(filename, 'r') as json_file:
                loaded_data = json.load(json_file)

            self.create_model( 
                loaded_data['model_name'],
                loaded_data['model_filename'], 
                loaded_data['encoder_filename'], 
                loaded_data['model_train_dataset'], 
                loaded_data['model_test_dataset'], 
                loaded_data.get('yolo_dataset_path', ''), # Use .get for backward compatibility
                loaded_data.get('annotation_dataset_path', ''), # Use .get for backward compatibility
                loaded_data.get('detector_model_path', ''), # For detector model
                loaded_data['model_classes'], 
                loaded_data['train_epochs'], 
                loaded_data['image_height'], 
                loaded_data['image_width']
                )
            return True

        except FileNotFoundError:
            print(f"Error: The file '{filename}' was not found.")
            return False
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from the file '{filename}'. Check for malformed data.")        
            return False
        except Exception:
            return False