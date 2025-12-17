#----- Application ------#
app_title = "Chassi Reader"
app_theme = "light"
app_wnd_mode = "Maximized"
app_wnd_width = 800
app_wnd_height = 800
app_customer_logo = ":images/SWIFT_BRANCO.png"
app_company_logo = ""


#----- Cameras ------#
cam_qty = 2
cam_usb_index = [ 0, 1]
cam_usb_color = [ 1, 1]
cam_usb_flip = [ 1, 1]
cam_auto_exposure = [ 0, 0] # 1 for auto, 0 for manual
cam_exposure = [ -16, -6]    # Manual exposure value
cam_auto_wb = [ 0, 0]       # 1 for auto, 0 for manual
cam_wb_temperature = [ 4000, 4000] # Manual white balance
cam_auto_focus = [ 0, 0]    # 1 for auto, 0 for manual
cam_focus = [ 255, 0]         # Manual focus value (0-255)

cam_files_path = "/Users/dayansantos/Dev/ocrChassi/models/sense_chassi_factory"
default_roi_x = 50
default_roi_y = 50
default_roi_w = 30
default_roi_h = 30





#----- Models ------#
model_train_epocs = 30
DATA_DIR = 'dataset' # Directory with your character subfolders
MODEL_SAVE_PATH = 'sensebike_ocr_model.keras'
LABEL_ENCODER_SAVE_PATH = 'sensebike_label_encoder.npy'
IMG_HEIGHT = 28
IMG_WIDTH = 28
EPOCHS = 15

#----- Segmentation -----#
segmentation_threshold = 50
segmentation_diameter = 9
segmentation_sigma = 75
segmentation_space = 75
segmentation_min_a = 0
segmentation_max_a = 100
segmentation_min_w = 3
segmentation_max_w = 100
segmentation_min_h = 1
segmentation_max_h = 100


#----- Programs ------#
program_path = ""





#----- Production ------#