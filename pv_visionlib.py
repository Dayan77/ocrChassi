import os
import ssl
#ssl._create_default_https_context = ssl._create_unverified_context

os.environ["KERAS_BACKEND"] = "tensorflow"

#import keras
#from keras import layers

#import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
import tensorflow_datasets as tfds
#import pytesseract
#import easyocr

import pyqtgraph as pg

import config_ini

import cv2

from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt


class pvVisionLib():
    
    def test_pytesseract(self, img):
        image_path = img#"/mnt/data/08d702e9-fb88-4a5b-98ce-9ad0bc19afd9.png"
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("Grayscale Image:")
        cv2.imshow('test',image_rgb)

        extracted_text = pytesseract.image_to_string(image_rgb)
        print(" Extracted Text:\n")
        print(extracted_text)

    def test_easyocr(self, img):
        reader = easyocr.Reader(['en'])
        extract_info = reader.readtext(img)

        for el in extract_info:
            print(el)

    def load_image_file(self, path):
        """Loads and preprocesses an image for segmentation."""
        img = cv2.imread(path)
        
        return img
    
    def convert_qt_image(self, img):
        if len(img.shape) >2:
            h, w, ch = img.shape
        else:
            h, w = img.shape
            ch = 2

        qt_image = None
        if ch > 2:
            qt_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            qt_image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        h, w, ch = qt_image.shape
        bytes_per_line = ch * w

        qt_image = QImage(qt_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        return pixmap

    def load_opencv_image(self, img):
        """Loads an image from a file using OpenCV and displays it."""
        try:
            
            imv = pg.ImageView()
            # Convert BGR to RGB
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            return img.transpose(1, 0, 2)
        
            # Display the image in the ImageView
            imv.setImage(img.transpose(1, 0, 2))
            
            # Set the view to auto-range
            imv.autoRange()

        except FileNotFoundError as e:
            print(e)
            # Create dummy image data if file is not found
            data = np.random.normal(size=(500, 500), loc=100, scale=50).astype(np.float32)
            imv.setImage(data)

        return imv

        
    def convert_to_gray(self, img):
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        return gray

    def convert_binary(self, gray, d, sigma, space, threshold):
        # Apply bilateral filter to reduce noise
        filtered = cv2.bilateralFilter(gray, d, sigma, space)
        # Binarize the image using Otsu's method
        _, binary = cv2.threshold(filtered, 0, threshold, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        return binary
        
    
    def find_characters(self, binary_image):
        """Finds and filters character components from a binary image."""
        # Find all connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, 8, cv2.CV_32S)
        
        characters = []
        # Loop over all components
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            # Filter out noise and too-large components
            if w > 5 and h > 10 and area > 50:
                characters.append((x, y, w, h))
                
        # Sort characters from left to right
        characters.sort(key=lambda char: char[0])
        return characters
    

    def collect_samples(self, image_path, output_dir, file_label, threshold, pixel, sigma, space, min_w, max_w, min_h, max_h, min_a, max_a):
        """Segments characters from an image, saves them, and requires manual labeling."""
        # Load and preprocess the original color image
        img = cv2.imread(image_path)
        gray = self.convert_to_gray(img)
        binary = self.convert_binary(gray, pixel, sigma, space, threshold)
        characters = self.find_characters(binary)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        boxes_img = self.draw_samples(binary, characters)
        samples = self.crop_save_samples(img, characters, "", file_label)

        return img, binary, boxes_img, samples, characters

        # cv2.imshow("Characters Detected", img)
        # cv2.imshow("Filtered Image", binary)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def draw_samples(self, img, characters):
        for i, (x, y, w, h) in enumerate(characters):
            # Crop the character from the original image
            char_image = img[y:y+h, x:x+w]
            
            # Displaying bounding boxes for verification (optional)
            #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        print(f"Boxes {len(characters)} character samples to image")
        return img


    def crop_save_samples(self, img, characters, output_dir, file_label):
        samples = {}
        for i, (x, y, w, h) in enumerate(characters):
            # Crop the character from the original image
            char_image = img[y:y+h, x:x+w]
            
            # Save the cropped image with a placeholder name
            # You must manually rename these to reflect the true character
            if output_dir != "":
                filename = f"{file_label}_char_{i}.png"
                cv2.imwrite(os.path.join(output_dir, filename), char_image)
            
            # Displaying bounding boxes for verification (optional)
            #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            samples.update({str(i):{ "char": "?", "box":{"x":x,"y":y,"w":w,"h":h}, "image":char_image} })

        print(f"Saved {len(characters)} character samples to {output_dir}")
        return samples
    
    
        
        
        
