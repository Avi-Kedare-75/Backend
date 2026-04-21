import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

IMG_SIZE = (224,224)
BATCH_SIZE = 16
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = train_datagen.flow_from_directory(
    "dataset/animals",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="training"
)

val_data = train_datagen.flow_from_directory(
    "dataset/animals",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation"
)
class_names = list(train_data.class_indices.keys())

with open("classes.json","w") as f:
    json.dump(class_names,f)

print("Classes saved:", class_names)

base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

for layer in base_model.layers:
    layer.trainable = False

    x = base_model.output

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dense(128, activation="relu")(x)

x = tf.keras.layers.Dropout(0.5)(x)

predictions = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)

model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)
model.save("animal_model.h5")

print("Model saved successfully")