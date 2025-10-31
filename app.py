# app.py
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the model
model = load_model('face_emotionModel.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Route for homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                img = Image.open(file).convert('L').resize((48,48))
                img_array = img_to_array(img)/255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array)
                emotion = emotion_labels[np.argmax(prediction)]
                result = f'Predicted Emotion: {emotion}'

                # Save output to database.txt
                with open('database.txt', 'a') as db:
                    db.write(f'User uploaded: {file.filename}, Prediction: {emotion}\n')

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
