# Flask ML API

A simple Flask API for training, testing, and predicting using a **Linear Regression model**.  
This API allows users to upload CSV or Excel files to train a model, test it, and make predictions on new data.

---
## 1️⃣ Recommended File Structure


flask-ml-app/           
│
├── app.py              <- Your main Flask app
├── requirements.txt    <- Python dependencies
├── README.md           <- Project description & usage
├── model.pkl           <- Trained ML model (generated after training)
├── encoder_columns.pkl <- Saved encoder columns (generated after training)
│
└── data/        <- Optional folder for sample CSV/XLSX files
    └── house_prices_pune_real.csv      <- for train (use any file ,its a eg file of my project)
    └── house_prices_pune_test_50.csv   <- for testing 
## Table of Contents

1. [Features](#features)  
2. [Installation](#installation)  
3. [Usage](#usage)  
4. [API Endpoints](#api-endpoints)  
5. [Using Postman](#using-postman)  
6. [Notes](#notes)  

---

## Features

- Train a Linear Regression model using uploaded CSV/Excel files (`/train`)  
- Test the model with new datasets (`/test`)  
- Predict values for a single sample via JSON (`/predict`)  
- Automatic handling of categorical data via one-hot encoding (`area` column)  
- Saves trained model and encoder columns for later use (`model.pkl`, `encoder_columns.pkl`)  

---
# files for train and test :
  EXAMPLES OF FILES FOR TRAIN MODEL:- given in file stucture as(house_prices_pune_real)
  AND FOR TESTING THE MODEL  :-  given in file stucture as(house_prices_pune_test_50)

 ## Installation

1. Clone the repository:


git clone https://github.com/yourusername/flask-ml-app.git
cd flask-ml-app

2.Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

3.Install dependencies:

pip install -r requirements.txt
 

---

##  Usage

Run the Flask app:

python app.py


By default, the API will run on:

http://127.0.0.1:5000/

API Endpoints:

1️⃣ Train Model

URL: /train

Method: POST

Form Data:

file → Upload CSV or Excel file with features and target column

target → Name of the target column

Response:
    {
    "message": "Model trained successfully",
    "mse": 123.45,
    "r2": 0.87
}


2️⃣ Test Model

URL: /test

Method: POST

Form Data:

file → CSV/Excel file with features and target

target → Target column name

Response:
    {
    "message": "Model tested successfully",
    "mse": 110.23,
    "r2": 0.90
}


3️⃣ Predict

URL: /predict

Method: POST

JSON Body: Include all feature columns

  {
  "sqft": 1500,
  "bath":2 ,
  "bhk": 1,
  "area": "Katraj"
}

Response:
  {
    "prediction": 12563429.96
}

---

## Using Postman

You can test all API endpoints using Postman:

Train Endpoint

Method: POST

URL: http://127.0.0.1:5000/train

* Body → Form-Data:

* Key: file → Choose a CSV/XLSX file

* Key: target → Enter target column name

Send the request and check the response for mse and r2.

Test Endpoint

Method: POST

URL: http://127.0.0.1:5000/test

* Body → Form-Data:

* Key: file → CSV/XLSX test file

* Key: target → Target column name

Send request → Response contains test metrics.

Predict Endpoint

Method: POST

URL: http://127.0.0.1:5000/predict

* Body → raw JSON:

  {
  "sqft": 1500,
  "bath":2 ,
  "bhk": 1,
  "area": "Katraj"
  }


Send request → Response contains predicted value.

✅ Tip: Always include all columns that were used during training, especially one-hot encoded categorical columns.

Notes

After training, model.pkl and encoder_columns.pkl are created. Keep them safe for later predictions.

If your dataset has new categorical values, ensure to match the one-hot encoding structure.

For large Excel files, openpyxl library is required (already included in requirements.txt).
