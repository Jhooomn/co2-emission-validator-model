"""
@author: Jhon Baron
"""
import numpy as np
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
loaded_model = joblib.load("reg_model.joblib")


@app.route('/predict/<cylynder_size>',methods=['GET'])
def predict(cylynder_size):
    print("VAL: ", cylynder_size)
    prediction = loaded_model.predict([[float(cylynder_size)]])
    output = prediction[0][0]
    return jsonify(result=output)


if __name__ == "__main__":
    app.run(debug=True)