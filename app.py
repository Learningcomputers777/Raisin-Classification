import numpy as np
import os

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from features import process_image
from predictor import predict

app = Flask(__name__)
# Set the path to the folder for uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('input.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' in request.files:
        image_file = request.files['image']
        if image_file and allowed_file(image_file.filename):
            # Save the uploaded file to a folder
            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
            image_file.save(file_path)

        # Do something with the uploaded image, e.g., save it to a folder
        area, major_axis_length, minor_axis_length, eccentricity, convex_area, extent, perimeter, image = process_image(file_path)
        user_features  = np.asarray([area, major_axis_length, minor_axis_length, eccentricity, convex_area, extent, perimeter])
        result = predict(user_features)
        return f"Raisin class: {result}"
    else:
        return "No image provided in the request."

if __name__ == '__main__':
    app.run(debug=True)
