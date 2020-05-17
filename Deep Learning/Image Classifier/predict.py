import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import numpy as np
import json
import argparse
from PIL import Image


def process_image(image):
  tf_image = tf.convert_to_tensor(image)
  tf_image = tf.cast(tf_image, tf.float32)
  tf_image = tf.image.resize(tf_image, (224,224))
  tf_image /= 255
  
  return tf_image.numpy()

def get_model(model_path):
    my_model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    
    return my_model

def predict(image_path, model, k, category_names):
  img = Image.open(image_path)
  test_image = np.asarray(img)
  processed_test_image = process_image(test_image)
  processed_test_image = np.expand_dims(processed_test_image,0)
  #get classes
  with open(category_names, 'r') as f:
    class_names = json.load(f)
    
  #predicting the image
  p = model.predict(processed_test_image)

  probs, classes = tf.math.top_k(p, k)
  classes += 1
  top_prediction = class_names[str((np.array(classes)[0])[0])]

  return list(np.array(probs)[0]), list(np.array(classes)[0]), top_prediction
    
    
def main():
    #initialize the parser
    parser = argparse.ArgumentParser(description='to get the prediction type: python predict.py <image_path> <model_path> --top_k <k> --category_names <label_path>')

    #Add the positional parameters
    parser.add_argument('image', help='Path to the image', type = str)
    parser.add_argument('model', help='Path to model.h5', type=str)
    #Add the optional parameters
    parser.add_argument('--top_k', help='Top k predictions', type=int, default=5)
    parser.add_argument('--category_names', help='path to labels map', type=str, default='label_map.json')

    #Parse the argument
    args = parser.parse_args()
    #get the model from the model path
    my_model = get_model(args.model)
    
    #predict the image
    prob, classes, prediction = predict(args.image, my_model, args.top_k, args.category_names)
    
    print('prediction: ',prediction)
    print('top {} probabilities: {}'.format(args.top_k, prob))
    print('top {} classes: {}'.format(args.top_k, classes))

if __name__ == '__main__':
    main()