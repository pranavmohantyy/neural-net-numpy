import numpy as np
import requests
import gzip
import os

def download_mnist():
    url = 'http://yann.lecun.com/exdb/mnist/'
    os.makedirs('data', exist_ok=True)
    for filename in ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']:
        response = requests.get(url + filename)
        with open(os.path.join('data', filename), 'wb') as f:
            f.write(response.content)

    return 'data'

def load_mnist():
    base_path = download_mnist()
    with gzip.open(os.path.join(base_path, 'train-images-idx3-ubyte.gz'), 'rb') as f:
        X_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28) / 255.0
    with gzip.open(os.path.join(base_path, 'train-labels-idx1-ubyte.gz'), 'rb') as f:
        y_train = np.frombuffer(f.read(), np.uint8, offset=8)
    return X_train, y_train