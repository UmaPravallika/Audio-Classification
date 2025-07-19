import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import librosa 

app = Flask(__name__)
UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Class labels
class_names = ['dog_bark', 'children_playing', 'car_horn', 'air_conditioner',
               'street_music', 'gun_shot', 'siren', 'engine_idling', 'jackhammer',
               'drilling']

# Load model and encoder
model = load_model("model.h5")
labelencoder = LabelEncoder()
labelencoder.fit(class_names)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    if "audio" not in request.files:
        return "Audio file not found", 400

    file = request.files["audio"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Extract features using librosa
    audio, sr = librosa.load(filepath, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    input_data = mfcc_scaled.reshape(1, -1)

    # Prediction
    prediction = model.predict(input_data)
    predicted_class_index = np.argmax(prediction)
    confidence = round(float(np.max(prediction)) * 100, 2)
    predicted_label = labelencoder.inverse_transform([predicted_class_index])[0]

    return render_template("result.html",
                           prediction=predicted_label,
                           confidence=confidence,
                           audio_file=file.filename)

if __name__ == "__main__":
    app.run(debug=True)
