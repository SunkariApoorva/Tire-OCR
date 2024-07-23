import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from google.cloud import vision
from google.cloud.vision_v1 import types
import io

# Set the Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\sunka\Downloads\sonic-ego-426808-a7-ddc5fa32cf43.json"

def perform_ocr(image_path):
    """Perform OCR using Google Cloud Vision API"""
    client = vision.ImageAnnotatorClient()
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = types.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description if texts else "No text detected."

# Define paths to your dataset
train_data_dir = 'train'
validation_data_dir = 'valid'
image_folder = 'images'  # Folder containing the images for inference

# Create DataFrames for train and validation data
def get_image_files(data_dir):
    return [os.path.join(data_dir, img) for img in os.listdir(data_dir) if img.endswith(('jpg', 'jpeg'))]

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

# Load EfficientNet B0 pre-trained on ImageNet
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))

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

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input)

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
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

# First Training Phase
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[checkpoint, early_stopping])

# Optionally, unfreeze some layers and continue training
for layer in model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Second Training Phase
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[checkpoint, early_stopping])

# Save the final model
model.save('tire_detection_model.h5')

# Inference code
def detect_tire_and_extract_text(image_path, model):
    # Load the image
    image = cv2.imread(image_path)
    orig_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    # Predict if the image contains a tire
    prediction = model.predict(image)[0][0]

    if prediction > 0.5:
        print(f"Tire detected in {image_path}")

        # Proceed to extract text
        extracted_text = perform_ocr(image_path)
        return extracted_text
    return "No tire detected."

# Load the trained model
trained_model = load_model('tire_detection_model.h5')

# Open a file to write the results
with open('extracted_text.txt', 'w', encoding='utf-8') as f:
    # Process images from the folder
    for img_file in os.listdir(image_folder):
        if img_file.endswith(('jpg', 'jpeg')):
            img_path = os.path.join(image_folder, img_file)
            extracted_text = detect_tire_and_extract_text(img_path, trained_model)
            f.write(f"Image: {img_file}, Extracted Text: {extracted_text}\n")
