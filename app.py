import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
   float_attributes = [float(x) for x in request.form.values()]
   final_attributes = [np.array(float_attributes)]
   prediction = model.predict(final_attributes)
   return render_template("index.html", prediction_text = "Based on the information you entered, there is a likelihood of being sent {}".format(prediction))


if __name__ == "__main__":
    app.run(port=3000)
