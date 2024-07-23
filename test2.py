import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import keras_ocr
import pytesseract
import requests
import json
from google.cloud import vision
import io

# Set the Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\sunka\Downloads\sonic-ego-426808-a7-ddc5fa32cf43.json"

# Set the API keys and model IDs for NanoNets
os.environ["NANONETS_API_KEY"] = "cd18ec99-4816-11ef-9e5e-ea8efc3630fe"
os.environ["NANONETS_MODEL_ID"] = "128a2d11-9f96-4d62-af62-29c792fedaf5"

# Define paths and directories
train_data_dir = 'train'
validation_data_dir = 'valid'
image_folder = 'images'  # Folder containing the images for inference

# Create DataFrames for train and validation data
def get_image_files(data_dir):
    return [os.path.join(data_dir, img) for img in os.listdir(data_dir) if img.endswith(('jpg', 'jpeg', 'png'))]

train_images = get_image_files(train_data_dir)
validation_images = get_image_files(validation_data_dir)

def create_dataframe(images, label):
    return pd.DataFrame({'filename': images, 'label': label})

train_df = create_dataframe(train_images, 'tire')
validation_df = create_dataframe(validation_images, 'tire')

# Add dummy second class
dummy_train_df = create_dataframe(train_images[:10], 'non_tire')  # Reuse some images for dummy class
dummy_validation_df = create_dataframe(validation_images[:5], 'non_tire')  # Reuse some images for dummy class

# Combine DataFrames
train_df = pd.concat([train_df, dummy_train_df])
validation_df = pd.concat([validation_df, dummy_validation_df])

# Parameters
batch_size = 32
epochs = 20
image_size = (224, 224)

# Register the swish activation function
tf.keras.utils.get_custom_objects().update({'swish': tf.keras.activations.swish})

# Load EfficientNet B3 pre-trained on ImageNet
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))

# Add custom layers for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Create model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base layers during initial training
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Data augmentation including rotation and brightness adjustments
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    brightness_range=[0.8, 1.2],
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)

# Debug: Check if data generators found images
print(f"Found {len(train_df)} images in training directory.")
print(f"Found {len(validation_df)} images in validation directory.")

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='label',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_dataframe(
    validation_df,
    x_col='filename',
    y_col='label',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary')

# Calculate steps_per_epoch and validation_steps
steps_per_epoch = max(1, len(train_df) // batch_size)
validation_steps = max(1, len(validation_df) // batch_size)

# Callbacks
checkpoint = ModelCheckpoint('tire_detection_model_new.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

# Train the new model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[checkpoint, early_stopping]
)

# Save the final model
model.save('tire_detection_model_new.keras')

def load_model():
    """Load the model from a file"""
    try:
        model = tf.keras.models.load_model('tire_detection_model_new.keras', custom_objects={'swish': tf.keras.activations.swish})
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Function to perform OCR using NanoNets
def perform_ocr_nanonets(image_path):
    """Perform OCR using NanoNets API"""
    url = 'https://app.nanonets.com/api/v2/OCR/Model/' + os.environ["NANONETS_MODEL_ID"] + '/LabelFile/'
    data = {'file': open(image_path, 'rb')}
    response = requests.post(url, auth=requests.auth.HTTPBasicAuth(os.environ["NANONETS_API_KEY"], ''), files=data)
    result = response.json()
    if 'result' in result:
        return " ".join([obj['text'] for obj in result['result'][0]['prediction']])
    return "No text detected."

# Function to perform OCR using Google Vision
def perform_ocr_google_vision(image_path):
    """Perform OCR using Google Cloud Vision API"""
    client = vision.ImageAnnotatorClient()
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description.replace('\n', ' ') if texts else "No text detected."

# Function to perform OCR using pytesseract
def perform_ocr_pytesseract(image_path):
    """Perform OCR using pytesseract"""
    text = pytesseract.image_to_string(image_path)
    return text.replace('\n', ' ')

# Function to perform OCR using keras-ocr
def perform_ocr_keras_ocr(image_path):
    """Perform OCR using keras-ocr"""
    pipeline = keras_ocr.pipeline.Pipeline()
    images = [keras_ocr.tools.read(image_path)]
    prediction_groups = pipeline.recognize(images)
    text = " ".join([word for word, box in prediction_groups[0]])
    return text

# Function to detect tire and extract text
def detect_tire_and_extract_text(image_path, model):
    # Load the image
    image = cv2.imread(image_path)
    orig_image = image.copy()
    
    # Resize and preprocess the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    image = preprocess_input(np.expand_dims(image, axis=0))  # Ensure correct shape

    # Predict if the image contains a tire
    prediction = model.predict(image)[0][0]

    # Perform OCR using four different methods
    text_google_vision = perform_ocr_google_vision(image_path)
    text_pytesseract = perform_ocr_pytesseract(image_path)
    text_keras_ocr = perform_ocr_keras_ocr(image_path)
    text_nanonets = perform_ocr_nanonets(image_path)

    # Combine texts from different methods
    combined_text = f"Google Vision OCR: {text_google_vision}\n" \
                    f"Pytesseract OCR: {text_pytesseract}\n" \
                    f"Keras OCR: {text_keras_ocr}\n" \
                    f"NanoNets OCR: {text_nanonets}"
    
    return prediction, combined_text

# Load the model
model = load_model()

# Process all images in the image folder
if model:
    image_files = get_image_files(image_folder)
    for image_path in image_files:
        prediction, combined_text = detect_tire_and_extract_text(image_path, model)
        print(f"Image: {image_path}")
        print(f"Prediction: {'Tire' if prediction >= 0.5 else 'Non-Tire'}")
        print(f"Extracted Text:\n{combined_text}\n")
else:
    print("Model could not be loaded.")
