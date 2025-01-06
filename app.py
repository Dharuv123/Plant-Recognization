import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template_string, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from tensorflow.keras.models import load_model
from PIL import Image

# Directory settings
UPLOAD_FOLDER = 'static/upload'  # Move upload folder to static for Flask to serve files
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# MongoDB setup
client = MongoClient('localhost', 27017)  # Connect to MongoDB locally
db = client.plant  # Database name
collection = db.values  # Collection name

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the Keras model and labels
MODEL_PATH = "tf_files/retrained_graph.h5"
LABELS_PATH = "tf_files/retrained_labels.txt"

model = load_model(MODEL_PATH)

with open(LABELS_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for uploading image
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the POST request has a file
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Save the file if it's valid
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('classification_process', filepath=filepath))
    
    # HTML content for the upload page
    upload_html = '''
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Upload Image for Plant Classification</title>
      <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/css/bootstrap.min.css">
      <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        html, body { height: 100%; width: 100%; font-family: 'Roboto', sans-serif; }
        body { background-image: url('https://www.w3schools.com/w3images/forest.jpg'); background-size: cover; background-position: center; background-attachment: fixed; color: white; }
        .overlay { background-color: rgba(0, 0, 0, 0.5); position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: -1; }
        .container { background-color: rgba(255, 255, 255, 0.8); border-radius: 8px; padding: 40px; box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2); max-width: 500px; margin-top: 100px; width: 80%; }
        h1 { font-size: 2rem; color: #2a3d4f; margin-bottom: 20px; }
        .btn-primary { background-color: #007bff; border: none; padding: 12px 25px; font-size: 1rem; border-radius: 8px; transition: background-color 0.3s ease; }
        .btn-primary:hover { background-color: #0056b3; }
        .form-control { padding: 15px; font-size: 1.1rem; border-radius: 8px; }
        .form-label { font-weight: bold; font-size: 1.1rem; margin-bottom: 15px; color : #000; }
      </style>
    </head>
    <body>
      <div class="overlay"></div>
      <div class="container">
        <h1 class="text-center">Upload Image of Plant</h1>
        <form action="/" method="post" enctype="multipart/form-data">
          <div class="mb-4">
            <label for="file" class="form-label">Choose an Image</label>
            <input type="file" name="file" class="form-control" id="file" required>
          </div>
          <div class="text-center">
            <button type="submit" class="btn btn-primary">Upload</button>
          </div>
        </form>
      </div>
    </body>
    </html>
    '''
    return render_template_string(upload_html)

# Route for classification
@app.route('/classify/<path:filepath>')
def classification_process(filepath):
    try:
        # Preprocess the image
        image = Image.open(filepath).convert('RGB')
        image = image.resize((224, 224))  # Adjust size based on model's requirement
        image_array = np.array(image) / 255.0  # Normalize to [0, 1]
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Get predictions
        predictions = model.predict(image_array)
        top_idx = np.argmax(predictions[0])
        confidence = predictions[0][top_idx]

        # Define a threshold for confidence
        CONFIDENCE_THRESHOLD = 0.6
        if confidence < CONFIDENCE_THRESHOLD:
            # If confidence is low, mark it as "Unknown Plant"
            plant_name = "Unknown Plant"
            return redirect(url_for('show_output_page', plant=plant_name, confidence=confidence,
                                   botanical_name='N/A', chemical_components='N/A',
                                   medicinal_properties='N/A', medical_uses='N/A',
                                   image_filename=os.path.basename(filepath)))

        # Get the plant name from labels
        plant_name = labels[top_idx]
        filename = os.path.basename(filepath)

        # Fetch additional data from MongoDB based on the plant name
        plant_info = collection.find_one({"Plant Name": plant_name})
        if plant_info:
            botanical_name = plant_info.get('Botanical Name', 'N/A')
            chemical_components = plant_info.get('Chemical Components', 'N/A')
            medicinal_properties = plant_info.get('Medicinal Properties', 'N/A')
            medical_uses = plant_info.get('Medical Uses', 'N/A')
        else:
            # Log warning for unrecognized plant
            print(f"Warning: Plant '{plant_name}' not found in the database.")
            botanical_name = 'N/A'
            chemical_components = 'N/A'
            medicinal_properties = 'N/A'
            medical_uses = 'N/A'

        return redirect(url_for('show_output_page', plant=plant_name, confidence=confidence,
                               botanical_name=botanical_name, chemical_components=chemical_components,
                               medicinal_properties=medicinal_properties, medical_uses=medical_uses,
                               image_filename=filename))

    except Exception as e:
        print(f"Error during classification: {e}")
        return jsonify({"error": f"Classification failed: {str(e)}"}), 500

# Route to display the result
@app.route('/Output/<plant>/<confidence>/<botanical_name>/<chemical_components>/<medicinal_properties>/<medical_uses>/<image_filename>')
def show_output_page(plant, confidence, botanical_name, chemical_components, medicinal_properties, medical_uses, image_filename):
    return render_template_string(
        '''
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>Plant Classification Result</title>
          <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/css/bootstrap.min.css">
          <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            html, body { height: 100%; width: 100%; font-family: 'Roboto', sans-serif; }
            body { background-image: url('https://www.w3schools.com/w3images/forest.jpg'); background-size: cover; background-position: center; background-attachment: fixed; color: white; }
            .overlay { background-color: rgba(0, 0, 0, 0.5); position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: -1; }
            .container { background-color: rgba(255, 255, 255, 0.8); border-radius: 8px; padding: 40px; box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2); max-width: 800px; margin-top: 50px; width: 90%; }
            h1 { font-size: 2rem; color: #2a3d4f; margin-bottom: 20px; }
            .img-fluid { border-radius: 10px; box-shadow: 0 5px 10px rgba(0, 0, 0, 0.1); margin-bottom: 20px; }
            .result-text { font-size: 1.2rem; color: #333; margin-bottom: 15px; }
            .result-heading { font-weight: bold; }
            .btn-back { background-color: #007bff; color: white; border: none; padding: 12px 25px; font-size: 1.1rem; border-radius: 8px; }
            .btn-back:hover { background-color: #0056b3; }
          </style>
        </head>
        <body>
          <div class="overlay"></div>
          <div class="container">
            <h1 class="text-center">Plant Classification Result</h1>
            <div class="text-center">
              <img src="{{ image_url }}" alt="Plant Image" class="img-fluid" width="300">
            </div>
            <div class="result-text">
              <span class="result-heading">Plant Name: </span>{{ plant }}
            </div>
            <div class="result-text">
              <span class="result-heading">Botanical Name: </span>{{ botanical_name }}
            </div>
            <div class="result-text">
              <span class="result-heading">Chemical Components: </span>{{ chemical_components }}
            </div>
            <div class="result-text">
              <span class="result-heading">Medicinal Properties: </span>{{ medicinal_properties }}
            </div>
            <div class="result-text">
              <span class="result-heading">Medical Uses: </span>{{ medical_uses }}
            </div>
            <div class="result-text confidence-text">
              Confidence: {{ confidence }}%
            </div>
            <div class="text-center">
              <a href="/" class="btn btn-back">Back to Home</a>
            </div>
          </div>
        </body>
        </html>
        ''',
        plant=plant,
        confidence=f"{float(confidence) * 100:.2f}",
        botanical_name=botanical_name,
        chemical_components=chemical_components,
        medicinal_properties=medicinal_properties,
        medical_uses=medical_uses,
        image_url=url_for('static', filename=f'upload/{image_filename}')
    )

if __name__ == '__main__':
    app.run(debug=True)
