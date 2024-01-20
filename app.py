# Step1: 
# Import Packages
import numpy as np
from flask import Flask, request, render_template
import pickle

# Step2:
# Create an app object using the Flask class. 
app = Flask(__name__)
# Load the trained model. (Pickle file)
model = pickle.load(open('models/model.pkl', 'rb'))

# Step3:
# use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    return render_template('index.html')

# Step4:
# Redirect to /predict page with the output
@app.route('/predict',methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "The flower species is {}".format(prediction))

# Step5:
# Run the App
if __name__ == "__main__":
    app.run(debug=True)
