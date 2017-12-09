import os
import numpy as np
import pandas as pd
from scipy.misc import imread
import tensorflow as tf

from six.moves import urllib

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, InputLayer
from keras.regularizers import L1L2
from scipy.misc import imsave

import gzip
import os
import sys
import time
import csv

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10


SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
print('Hi')
def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].
  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    #data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
    return data

def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  return labels

def maybe_download(filename):
  """Download the data from Yann's website, unless it's already here."""
  root_dir = os.path.abspath('.')
  data_dir = os.path.join(root_dir, 'Data')
  WORK_DIRECTORY =  data_dir
  print('a')
  if not tf.gfile.Exists(WORK_DIRECTORY):
    print('b')
    tf.gfile.MakeDirs(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not tf.gfile.Exists(filepath):
    print('c')
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
  print('d')
  return filepath

# to stop potential randomness
seed = 128
rng = np.random.RandomState(seed)


# set path
root_dir = os.path.abspath('.')
data_dir = os.path.join(root_dir, 'Data')
print('data dir')
print(data_dir)

train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')
print('e')
print(train_data_filename)
train_data = extract_data(train_data_filename, 60000)
train_labels = extract_labels(train_labels_filename, 60000)
test_data = extract_data(test_data_filename, 10000)
test_labels = extract_labels(test_labels_filename, 10000)


if not os.path.isdir("mnist/train-images"):
   os.makedirs("mnist/train-images")

if not os.path.isdir("mnist/test-images"):
   os.makedirs("mnist/test-images")


# process train data
with open("mnist/train-labels.csv", 'w') as csvFile:
  writer = csv.writer(csvFile, delimiter=',', quotechar='"')
  for i in range(len(train_data)):
    imsave("mnist/train-images/" + str(i) + ".jpg", train_data[i][:,:,0])
    writer.writerow(["train-images/" + str(i) + ".jpg", train_labels[i]])

# repeat for test data
with open("mnist/test-labels.csv", 'w') as csvFile:
  writer = csv.writer(csvFile, delimiter=',', quotechar='"')
  for i in range(len(test_data)):
    imsave("mnist/test-images/" + str(i) + ".jpg", test_data[i][:,:,0])
    writer.writerow(["test-images/" + str(i) + ".jpg", test_labels[i]])

# load data
train = pd.read_csv(os.path.join('D:\\gan\\mnist','train-labels.csv'))
test = pd.read_csv(os.path.join('D:\\gan\\mnist', 'test-labels.csv'))
print('ds')
print(train)
temp = []
for index,row in train.iterrows():
    print("heres")
    print(row)
    print("dg")
    print(row[0])
    print("ddg")
    print(row[1])
    image_path = os.path.join(data_dir, 'train-images', img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)

train_x = np.stack(temp)

train_x = train_x / 255.

# print image
img_name = rng.choice(train.filename)
image_path = os.path.join(data_dir, 'train-images', img_name)

img = imread(filepath, flatten=True)

pylab.imshow(img, cmap='gray')
pylab.axis('off')
pylab.show()



g_input_shape = 100
d_input_shape = (28, 28)
hidden_1_num_units = 500
hidden_2_num_units = 500
g_output_num_units = 784
d_output_num_units = 1
epochs = 25
batch_size = 128

# generator
model_1 = Sequential([
    Dense(units=hidden_1_num_units, input_dim=g_input_shape, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)),

    Dense(units=hidden_2_num_units, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)),

    Dense(units=g_output_num_units, activation='sigmoid', kernel_regularizer=L1L2(1e-5, 1e-5)),

    Reshape(d_input_shape),
])

# discriminator
model_2 = Sequential([
    InputLayer(input_shape=d_input_shape),

    Flatten(),

    Dense(units=hidden_1_num_units, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)),

    Dense(units=hidden_2_num_units, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)),

    Dense(units=d_output_num_units, activation='sigmoid', kernel_regularizer=L1L2(1e-5, 1e-5)),
])


from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling


gan = simple_gan(model_1, model_2, normal_latent_sampling((100,)))
model = AdversarialModel(base_model=gan,player_params=[model_1.trainable_weights, model_2.trainable_weights])
model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(), player_optimizers=['adam', 'adam'], loss='binary_crossentropy')

history = model.fit(x=train_x, y=gan_targets(train_x.shape[0]), epochs=10, batch_size=batch_size)

plt.plot(history.history['player_0_loss'])
plt.plot(history.history['player_1_loss'])
plt.plot(history.history['loss'])


zsamples = np.random.normal(size=(10, 100))
pred = model_1.predict(zsamples)
for i in range(pred.shape[0]):
    plt.imshow(pred[i, :], cmap='gray')
    plt.show()
