from flask import Flask, render_template, request
import os
import numpy as np
import joblib

app = Flask(__name__)
filename = 'file_iris1.pkl'
model = joblib.load(filename)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return {'status': 'ok'}, 200


@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    pred = model.predict(np.array([[sepal_length, sepal_width, petal_length, petal_width]]))
    return render_template('index.html', predict=str(pred))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', '5000')), debug=False)
