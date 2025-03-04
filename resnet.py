import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50, EfficientNetB0, DenseNet121
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# Parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15


# Train Data Generator (With Augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load Data
train_generator = train_datagen.flow_from_directory("chest_xray/train/", target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode="binary")
val_generator = val_datagen.flow_from_directory("chest_xray/val/", target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode="binary")
test_generator = test_datagen.flow_from_directory("chest_xray/test/", target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode="binary")

# Compute Class Weights
y_train = train_generator.classes
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

#  Load Pretrained ResNet50
# base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
# base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

base_model.trainable = False  # Freeze initial layers

# Add Classification Head (Fix Applied)
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)  # ✅ Properly connected

model = Model(inputs=base_model.input, outputs=output)

for layer in model.layers[-20:]:  # Unfreeze last 20 layers
    layer.trainable = True

# for layer in model.layers[-50:]:  # Unfreeze last 20 layers
#     layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])
#  Compile Model
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#               loss="binary_crossentropy",
#               metrics=["accuracy"])

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1)

# Train Model
model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS, class_weight=class_weights_dict, callbacks=[early_stopping, lr_scheduler])

# Evaluate Model
loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save Model
model.save("pneumonia_detection_model_densenet_fine_tuning30_15.h5")

