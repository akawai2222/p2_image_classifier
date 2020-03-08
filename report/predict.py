import argparse
parser = argparse.ArgumentParser()
parser.add_argument('image_path') 
parser.add_argument('saved_keras_model_filepath')
parser.add_argument('json_file')
parser.add_argument('--top_k',type = int)

args = parser.parse_args()

import numpy as np
import json
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
with open(args.json_file, 'r') as f:
    class_names = json.load(f)
reloaded_keras_model = tf.keras.models.load_model(args.saved_keras_model_filepath, custom_objects={'KerasLayer':hub.KerasLayer})

reloaded_keras_model.summary()

def predict(image_path, model, top_k):
  im = Image.open(image_path)
  im = im.resize((224,224)) 
  test_image = np.asarray(im)
  test_image = np.expand_dims(test_image, axis = 0)
  prediction = model.predict(test_image)
  probs, classes = tf.math.top_k(prediction, k=top_k)
  return probs, classes+1

probs, classes = predict(args.image_path, reloaded_keras_model, args.top_k)
names = []
for i in range(args.top_k):
  names.append(class_names[classes.numpy()[0,i].astype(str)])
print(class_names[classes.numpy()[0,0].astype(str)])
print(probs.numpy()[0,0])
