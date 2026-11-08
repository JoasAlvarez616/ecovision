# train_model.py — Entrena MobileNetV2 con 4 clases
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

IMG = (224, 224)
BATCH = 32
NUM_CLASSES = 4
EPOCHS = 15

train_dg = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20, zoom_range=0.15, horizontal_flip=True,
    width_shift_range=0.1, height_shift_range=0.1, fill_mode='nearest'
)
val_dg = ImageDataGenerator(preprocessing_function=preprocess_input)

train = train_dg.flow_from_directory("dataset/train", target_size=IMG,
                                     batch_size=BATCH, class_mode='categorical')
val = val_dg.flow_from_directory("dataset/val", target_size=IMG,
                                 batch_size=BATCH, class_mode='categorical', shuffle=False)
test = val_dg.flow_from_directory("dataset/test", target_size=IMG,
                                  batch_size=BATCH, class_mode='categorical', shuffle=False)

print("Clases:", train.class_indices)

base = MobileNetV2(weights='imagenet', include_top=False,
                   input_tensor=Input(shape=(*IMG, 3)))
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
out = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(base.input, out)
model.compile(Adam(1e-4), 'categorical_crossentropy', metrics=['accuracy'])

model.fit(
    train, validation_data=val, epochs=EPOCHS,
    callbacks=[
        EarlyStopping(monitor='val_accuracy', patience=5,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint("model/ecovision.keras", monitor='val_accuracy',
                        save_best_only=True, verbose=1),
    ],
    verbose=1
)

print("\n=== Evaluacion en TEST independiente ===")
y_pred = np.argmax(model.predict(test), axis=1)
y_true = test.classes
print(classification_report(y_true, y_pred,
                            target_names=list(test.class_indices.keys())))
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred))