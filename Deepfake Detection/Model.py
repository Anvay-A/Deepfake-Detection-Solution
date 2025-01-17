#This file contains the code for the model that will be used to train the data.
#The model is a combination of a Convolutional Neural Network (CNN) and a Recurrent Neural Network (RNN).
#This model is not functional yet as the datasets are not connected to the model yet.
#This is the general code for the model and the functions will be updated once the datasets are connected to the model.

import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import cv2
from hachoir.parser import createParser
from hachoir.metadata import extractMetadata
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import subprocess

# Function to extract the frames from the video

def extract_frames(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_frames)
    frames = []

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
    cap.release()
    return np.array(frames)

# Function to extract the metadata from the video using Hachoir library

def extract_video_metadata(video_path):
    parser = createParser(video_path)
    if not parser:
        raise ValueError(f"Unable to parse the video: {video_path}")
    metadata = extractMetadata(parser)
    if not metadata:
        raise ValueError(f"No metadata found in the video: {video_path}")
    metadata_dict = {item.key: item.value for item in metadata.exportPlaintext()}
    return metadata_dict

# Function to encode the metadata
def encode_metadata(metadata_dict):
    keys = list(metadata_dict.keys())
    values = list(metadata_dict.values())
    encoder = LabelEncoder()
    encoded_values = encoder.fit_transform(values).astype(float)
    scaler = MinMaxScaler()
    encoded_values = scaler.fit_transform(encoded_values.reshape(-1, 1)).flatten()
    return encoded_values

# Function to preprocess the video data and metadata for the model training and testing 

def preprocess_dataset_with_metadata(dataset_path, num_frames=10):
    X_frames, X_metadata, y = [], [], []
    for label, folder in enumerate(["real", "deepfake"]):
        folder_path = os.path.join(dataset_path, folder)
        for video_file in os.listdir(folder_path):
            video_path = os.path.join(folder_path, video_file)

            # Extract frames from the video
            frames = extract_frames(video_path, num_frames=num_frames)
            if len(frames) != num_frames:
                continue

            # Extract and encode metadata from the video
            try:
                metadata_dict = extract_video_metadata(video_path)
                metadata_encoded = encode_metadata(metadata_dict)
            except Exception as e:
                print(f"Error processing metadata for {video_path}: {e}")
                continue

            X_frames.append(frames)
            X_metadata.append(metadata_encoded)
            y.append(label)

    return (np.array(X_frames), np.array(X_metadata)), np.array(y)


#Function to build the model using the MobileNet CNN and LSTM RNN

def build_model(input_shape_frames, input_shape_metadata):
    frames_input = Input(shape=input_shape_frames)
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    x_frames = TimeDistributed(base_model)(frames_input)
    x_frames = TimeDistributed(GlobalAveragePooling2D())(x_frames)
    x_frames = LSTM(64)(x_frames)

    metadata_input = Input(shape=input_shape_metadata)
    x_metadata = Dense(64, activation="relu")(metadata_input)

    combined = Concatenate()([x_frames, x_metadata])
    combined = Dense(128, activation="relu")(combined)
    outputs = Dense(2, activation="softmax")(combined)

    model = Model(inputs=[frames_input, metadata_input], outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

#Function to train the model using the training data and test the model using the testing data.

def train_model(dataset_path, num_frames=10):
    (X_frames, X_metadata), y = preprocess_dataset_with_metadata(dataset_path, num_frames=num_frames)
    X_frames = X_frames / 255.0

    input_shape_frames = (num_frames, 224, 224, 3)
    input_shape_metadata = (len(X_metadata[0]),)

    model = build_model(input_shape_frames, input_shape_metadata)

    X_train_frames, X_val_frames, X_train_metadata, X_val_metadata, y_train, y_val = train_test_split(
        X_frames, X_metadata, y, test_size=0.2, random_state=42
    )

    checkpoint = ModelCheckpoint("deepfake_model.h5", save_best_only=True, monitor="val_accuracy", mode="max")
    early_stopping = EarlyStopping(monitor="val_loss", patience=5)

    model.fit(
        [X_train_frames, X_train_metadata], y_train,
        validation_data=([X_val_frames, X_val_metadata], y_val),
        epochs=20,
        batch_size=8,
        callbacks=[checkpoint, early_stopping]
    )
    return model

# Code for API to predict the deepfake videos using the trained model in mobile devices.

app = FastAPI()

@app.post("/predict")
async def predict(video: UploadFile):
    video_path = os.path.join("uploads", video.filename)

    with open(video_path, "wb") as f:
        f.write(await video.read())

    try:
        frames = extract_frames(video_path, num_frames=10) / 255.0
        metadata_dict = extract_video_metadata(video_path)
        metadata_encoded = encode_metadata(metadata_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    frames = np.expand_dims(frames, axis=0)
    metadata_encoded = np.expand_dims(metadata_encoded, axis=0)

    interpreter = tf.lite.Interpreter(model_path="deepfake_model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], frames)
    interpreter.set_tensor(input_details[1]['index'], metadata_encoded)

    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]

    return JSONResponse({"real_confidence": float(prediction[0]), "deepfake_confidence": float(prediction[1])})

# Deployment of the api using uvicorn server.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)