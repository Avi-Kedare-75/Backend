from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import json

app = Flask(__name__)
CORS(app)

# Load trained model
model = tf.keras.models.load_model("animal_model.h5")

# Load class labels
with open("classes.json") as f:
    classes = json.load(f)


def preprocess_image(image):

    image = image.resize((224,224))
    image = np.array(image)

    if image.shape[-1] == 4:
        image = image[:,:,:3]

    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    return image


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    image = Image.open(file.stream)

    processed = preprocess_image(image)

    pred = model.predict(processed)

    index = np.argmax(pred)

    label = classes[index]

    confidence = float(pred[0][index])

    return jsonify({
        "species": label,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(port=5000, debug=True)