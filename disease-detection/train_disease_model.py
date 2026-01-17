import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ======================
# CONFIG
# ======================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5   # keep small for first run
DATASET_DIR = "plantvillage dataset/color"


# ======================
# DATA LOADER
# ======================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# ======================
# MODEL (Transfer Learning)
# ======================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False  # freeze base model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ======================
# COMPILE
# ======================
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ======================
# TRAIN
# ======================
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ======================
# SAVE MODEL
# ======================
model.save("disease_model.h5")

print("✅ Disease detection model trained and saved!")
