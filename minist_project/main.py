import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

import minist_project.cnn as cnn

x = tf.placeholder("float", [None, 784])
sess = tf.Session()

def convolutional(input):
    return cnn.load_model(input)

app = Flask(__name__)

@app.route('/api/mnist', methods=['post'])
def mnist():
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(28,28)
    output1 = list(convolutional(input).astype(np.float))
    output2 = list(convolutional(input).astype(np.float))
    return jsonify(results=[output1, output2])

@app.route('/')
def main():
    return render_template('index.html')

if __name__ == '__main__':
    app.debug = True
    app.run(host='127.0.0.1', port=8000)
