import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10  

# Data Generators (With Augmentation)
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

#Load Data
train_generator = train_datagen.flow_from_directory("chest_xray/train/", target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode="binary")
val_generator = val_datagen.flow_from_directory("chest_xray/val/", target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode="binary")
test_generator = test_datagen.flow_from_directory("chest_xray/test/", target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode="binary", shuffle=False)

#Compute Class Weights
y_train = train_generator.classes
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
print("Computed Class Weights:", class_weights_dict)

#Build a simple CNN model from scratch
baseline_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification (Normal vs Pneumonia)
])

# Compile the model
baseline_model.compile(
    optimizer=Adam(learning_rate=1e-4),  
    loss="binary_crossentropy",  
    metrics=["accuracy"]
)

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Train the model
history = baseline_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights_dict,  # Apply class weights
    callbacks=[early_stopping],
    verbose=1
)

# Save the trained model
baseline_model.save("pneumonia_detection_model_CNN.h5")

# Evaluate on test data
test_loss, test_acc = baseline_model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.2f}")