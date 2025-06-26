from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploaded_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and setup
model = load_model('model/rice_model.h5')
image_size = (224, 224)

# Rebuild label encoder
class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
label_encoder = LabelEncoder()
label_encoder.fit(class_names)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'No file uploaded!'
        file = request.files['image']
        if file.filename == '':
            return 'No file selected!'
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess and predict
            img = cv2.imread(file_path)
            img = cv2.resize(img, image_size)
            img_array = np.expand_dims(img, axis=0)
            prediction = model.predict(img_array)
            pred_index = np.argmax(prediction)
            predicted_label = label_encoder.inverse_transform([pred_index])[0]
            confidence = round(float(np.max(prediction)) * 100, 2)

            return render_template('result.html',
                                   image_url=file_path,
                                   predicted_label=predicted_label,
                                   confidence=confidence)
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
