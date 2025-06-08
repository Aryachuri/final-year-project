from flask import Flask, render_template, request, redirect, url_for,send_file,jsonify
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from skimage import measure
import io
import base64
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import LSTM, Dense #type: ignore
import os
import convert
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "2"
import tensorflow as tf
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
######Login and signup code########

@app.route("/")
def login():
    return render_template("login.html", result=None)

@app.route("/signup.html")
def signup():
    return render_template("signup.html", result=None)

@app.route("/login.html")
def signup1():
    return render_template("login.html", result=None)

###### Home page code ########

@app.route("/index.html")
def index():
    return render_template("index.html", result=None)

@app.route("/index.html")
def index2():
    return render_template("index.html", result=None)



# analysis section code:-

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_deforestation(past_path, current_path):
    past_image = cv2.imread(past_path, 0)  # Grayscale
    current_image = cv2.imread(current_path, 0)  # Grayscale

    if past_image is None or current_image is None:
        return None  # Return None if image loading fails

    # Resize images to match dimensions
    height, width = past_image.shape
    current_image = cv2.resize(current_image, (width, height))

    # Compute absolute difference
    diff = cv2.absdiff(past_image, current_image)

    # Apply threshold
    _, thresholded = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

    # Label connected components (deforested regions)
    labels = measure.label(thresholded, connectivity=2)

    # Count deforested regions
    num_deforestation_events = labels.max()

    return num_deforestation_events  # Return as integer



##### Features code ########



@app.route("/features.html")
def features():
    if request.method == 'POST':
        image_file = request.files.get('image')
        mask_file = request.files.get('mask')

        grayscale_paths = []

        for file in [image_file, mask_file]:
            if file and file.filename != '':
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)

                img = Image.open(filepath).convert('L')  # Convert to grayscale
                gray_path = os.path.join(UPLOAD_FOLDER, f"grayscale_{file.filename}")
                img.save(gray_path)
                grayscale_paths.append(gray_path)

        # Send the first grayscale image as a download
        if grayscale_paths:
            return send_file(grayscale_paths[0], as_attachment=True)
        

    return render_template("features.html", result=None)

###### Analysis code ##########

@app.route("/analysis.html", methods=["GET", "POST"])
def analysis():
    if request.method == "POST":
        # Get uploaded files
        past_file = request.files.get("past_image")
        current_file = request.files.get("current_image")

        if not past_file or not current_file:
            return render_template("analysis.html", result="Error: No files uploaded.")

        if not (allowed_file(past_file.filename) and allowed_file(current_file.filename)):
            return render_template("analysis.html", result="Error: Invalid file type. Use PNG, JPG, or JPEG.")

        # Save files
        past_path = os.path.join(app.config['UPLOAD_FOLDER'], past_file.filename)
        current_path = os.path.join(app.config['UPLOAD_FOLDER'], current_file.filename)
        past_file.save(past_path)
        current_file.save(current_path)

        # Process images
        result = detect_deforestation(past_path, current_path)

        if result is None:
            return render_template("analysis.html", result="Error: Could not process images.")

        return render_template("analysis.html", result=result, past_file=past_file.filename, current_file=current_file.filename)

    return render_template("analysis.html", result=None)






################################# prediction code 4 ######################################
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import LSTM, Dense        #type: ignore
import matplotlib.pyplot as plt
import io
import base64


# Load data and pre-process (This part should ideally be done once)
file_path = 'D:/excel files and csv files/Subnational 1 tree cover loss.csv'  # Adjust as needed
df = pd.read_csv(file_path)

# Get unique values for dropdowns
country_options = df['country'].unique().tolist()
subnational1_options = df['subnational1'].unique().tolist()
threshold_options = df['threshold'].unique().tolist()

# Define time series columns for tree cover loss
time_series_columns = [col for col in df.columns if "tc_loss_ha_" in col]

def prepare_data(series, n_steps):
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i:i + n_steps])
        y.append(series[i + n_steps])
    return np.array(X), np.array(y)

# Flask route for prediction
@app.route('/predict.html', methods=['GET', 'POST'])
def predict():
    prediction_graph = None

    if request.method == 'POST':
        selected_country = request.form.get('country')
        selected_subnational1 = request.form.get('subnational1')
        selected_threshold = int(request.form.get('threshold'))

        filtered_df = df[
            (df['country'] == selected_country) &
            (df['subnational1'] == selected_subnational1) &
            (df['threshold'] == selected_threshold)
        ]

        print(f"Shape of filtered_df: {filtered_df.shape}")
        print(f"First few rows of filtered_df:\n{filtered_df.head()}")

        if not filtered_df.empty:
            # Extracting the relevant columns for prediction
            series = filtered_df[time_series_columns].values.flatten()
            if len(series) < 10:  # Ensure we have enough data to use
                print("Not enough data for prediction.")
                prediction_graph = None
            else:
                scaler = MinMaxScaler(feature_range=(0, 1))
                series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

                n_steps = 10
                X, y = prepare_data(series_scaled, n_steps)
                X = X.reshape((X.shape[0], X.shape[1], 1))

                split = int(0.8 * len(X))
                X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

                model = Sequential([
                    LSTM(50, activation='relu', input_shape=(n_steps, 1)),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                model.fit(X_train, y_train, epochs=200, verbose=0)

                y_pred = model.predict(X_test, verbose=0)
                y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
                y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

                plt.figure(figsize=(10, 5))
                plt.plot(y_test, label='Actual', marker='o')
                plt.plot(y_pred, label='Predicted', linestyle='dashed', marker='x')
                plt.title(f'LSTM Prediction for {selected_subnational1}')
                plt.xlabel('Time Steps')
                plt.ylabel('Tree Cover Loss (ha)')
                plt.legend()

                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                prediction_graph = base64.b64encode(img.getvalue()).decode()
                plt.close()
        else:
            print("Filtered dataframe is empty. Cannot train the model.")
            prediction_graph = None

        return render_template('predict.html', 
                           country_options=country_options,
                           subnational1_options=subnational1_options,
                           threshold_options=threshold_options,
                           prediction_graph=prediction_graph)

    return render_template('predict.html', country_options=country_options,subnational1_options=subnational1_options,threshold_options=threshold_options,prediction_graph=prediction_graph)

################################## prediction code 4 ends here ############################
@app.route("/map.html")
def map():
    return render_template("map.html", result=None)


if __name__ == '__main__':
    app.run(debug=True)