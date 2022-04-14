from tqdm import tqdm
import gzip
import os
import shutil
import urllib.request

if not os.path.exists('data'):
    os.makedirs('data/')

# Binary Alphadigits 

if not os.path.exists('data/binaryalphadigs.mat'):
    url = 'https://cs.nyu.edu/~roweis/data/binaryalphadigs.mat'
    urllib.request.urlretrieve(url, 'data/binaryalphadigs.mat')

# MNIST

MNIST_URL = 'http://yann.lecun.com/exdb/mnist/'

archives = ['train-images-idx3-ubyte.gz',
            'train-labels-idx1-ubyte.gz',
            't10k-images-idx3-ubyte.gz',
            't10k-labels-idx1-ubyte.gz'
           ]

for archive in tqdm(archives):
    url = MNIST_URL + archive
    if not os.path.exists('data/' + archive):
        urllib.request.urlretrieve(url, 'data/' + archive)

for archive in tqdm(archives):
    with gzip.open('data/' + archive, 'rb') as f_in:
        filename = archive[:-3]
        with open('data/' + filename, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove('data/' + archive)