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
from google.cloud import vision
from google.cloud.vision_v1 import types
import io
import matplotlib.pyplot as plt
import keras_ocr
import pytesseract
from lime import lime_image
from skimage.segmentation import mark_boundaries
from concurrent.futures import ThreadPoolExecutor

# Set the Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\sunka\Downloads\sonic-ego-426808-a7-ddc5fa32cf43.json"

# Define OCR functions
def perform_ocr_google_vision(image_path):
    """Perform OCR using Google Cloud Vision API"""
    client = vision.ImageAnnotatorClient()
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = types.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description.replace('\n', ' ') if texts else "No text detected."

def preprocess_image(image_path):
    """Preprocess the image to enhance text for OCR"""
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised_image = cv2.fastNlMeansDenoising(binary_image, None, 30, 7, 21)
    return denoised_image

def perform_ocr_pytesseract(image_path):
    """Perform OCR using pytesseract with improved preprocessing"""
    preprocessed_image = preprocess_image(image_path)
    text = pytesseract.image_to_string(preprocessed_image)
    return text.replace('\n', ' ')

def perform_ocr_keras_ocr(image_path):
    """Perform OCR using keras-ocr"""
    pipeline = keras_ocr.pipeline.Pipeline()
    images = [keras_ocr.tools.read(image_path)]
    prediction_groups = pipeline.recognize(images)
    text = " ".join([word for word, box in prediction_groups[0]])
    return text

# Define paths and directories
train_data_dir = 'train'
validation_data_dir = 'valid'
image_folder = 'images'

def get_image_files(data_dir):
    return [os.path.join(data_dir, img) for img in os.listdir(data_dir) if img.endswith(('jpg', 'jpeg'))]

def create_dataframe(images, label):
    return pd.DataFrame({'filename': images, 'label': label})

train_images = get_image_files(train_data_dir)
validation_images = get_image_files(validation_data_dir)

train_df = create_dataframe(train_images, 'tire')
validation_df = create_dataframe(validation_images, 'tire')

dummy_train_df = create_dataframe(train_images[:10], 'non_tire')
dummy_validation_df = create_dataframe(validation_images[:5], 'non_tire')

train_df = pd.concat([train_df, dummy_train_df])
validation_df = pd.concat([validation_df, dummy_validation_df])

# Parameters
batch_size = 32
epochs = 20
image_size = (224, 224)

# Load EfficientNet B3 model
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

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

steps_per_epoch = max(1, len(train_df) // batch_size)
validation_steps = max(1, len(validation_df) // batch_size)

checkpoint = ModelCheckpoint('new_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[checkpoint, early_stopping])

model.save('tire_detection_model.keras')

def detect_tire_and_extract_text(image_path, model):
    # Load and preprocess image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, image_size)
    image_preprocessed = preprocess_input(np.expand_dims(image_resized, axis=0))

    # Predict if the image contains a tire
    prediction = model.predict(image_preprocessed)[0][0]

    # Perform OCR using multiple methods
    texts = {
        "google_vision": perform_ocr_google_vision(image_path),
        "pytesseract": perform_ocr_pytesseract(image_path),
        "keras_ocr": perform_ocr_keras_ocr(image_path)
    }

    combined_text = " ".join(texts.values())

    if prediction > 0.4:
        print(f"Tire detected in {image_path}")

        # Use LIME for explanation
        explainer = lime_image.LimeImageExplainer()

        def predict_fn(images):
            return model.predict(preprocess_input(images))

        explanation = explainer.explain_instance(image_resized.astype('double'),
                                                 predict_fn,
                                                 top_labels=1,
                                                 hide_color=0,
                                                 num_samples=1000)

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
        img_boundry = mark_boundaries(temp / 2 + 0.5, mask)

        lime_folder = 'lime_plots'
        os.makedirs(lime_folder, exist_ok=True)
        lime_plot_path = os.path.join(lime_folder, os.path.basename(image_path))
        img_boundry_normalized = (img_boundry - img_boundry.min()) / (img_boundry.max() - img_boundry.min())
        plt.imsave(lime_plot_path, img_boundry_normalized)

        # Extract DOT code and date code
        dot_code = None
        date_code = None
        combined_words = combined_text.split()
        for i, word in enumerate(combined_words):
            if word.upper() == "DOT" and i + 1 < len(combined_words):
                dot_code_parts = [word]
                next_word_index = i + 1
                while next_word_index < len(combined_words):
                    next_word = combined_words[next_word_index]
                    if any(char.isdigit() for char in next_word) and len(''.join(filter(str.isdigit, next_word))) >= 4:
                        dot_code_parts.append(next_word)
                        break
                    else:
                        dot_code_parts.append(next_word)
                        next_word_index += 1
                dot_code = ' '.join(dot_code_parts)
                break

        if dot_code:
            digits = ''.join(filter(str.isdigit, dot_code))
            if len(digits) >= 4:
                date_code = digits[-4:]

        return os.path.basename(image_path), combined_text.strip(), dot_code, date_code
    else:
        print(f"No tire detected in {image_path}")
        return os.path.basename(image_path), combined_text.strip(), None, None

def process_images_in_folder(folder_path, model):
    image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda img_path: detect_tire_and_extract_text(img_path, model), image_paths))
    
    return results

output_file = 'extracted_texts.txt'
results = process_images_in_folder(image_folder, model)

with open(output_file, 'w', encoding='utf-8') as f:
    for filename, extracted_text, dot_code, date_code in results:
        f.write(f"Extracted text for {filename}: {extracted_text}\n")
        if dot_code:
            f.write(f"DOT code for {filename}: DOT {dot_code}\n")
        if date_code:
            f.write(f"Manufacturing date: {date_code}\n")
