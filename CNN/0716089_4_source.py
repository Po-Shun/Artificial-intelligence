import json

def save_predictions(student_id, predictions):
  # Please use this function to generate 'XXXXXXX_4_result.json'
  # `predictions` is a list of int (0 or 1; fake=0 and real=1)
  # For example, `predictions[0]` is the prediction given "unknown/0000.jpg".
  # it will be 1 if your model think it is real, else 0 (fake).

  assert isinstance(student_id, str)
  assert isinstance(predictions, list)
  assert len(predictions) == 5350

  for y in predictions:
    assert y in (0, 1)

  with open('{}_4_result.json'.format(student_id), 'w') as f:
    json.dump(predictions, f)
# !wget https://lab.djosix.com/icons.zip -O /data/icons.zip
# import os 
# import zipfile
# local_zip = "/data/icons.zip"
# zip_ref = zipfile.ZipFile(local_zip,"r")
# zip_ref.extractall("/data")
# zip_ref.close()
import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
# import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
train = ImageDataGenerator(rescale = 1./255)
test = ImageDataGenerator(rescale=1/255)

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "data",
    validation_split=0.2,
    subset="training",
    seed=1337,
    label_mode = "binary",
    image_size=(64,64),
    batch_size=60,
)
                                         
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "data",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    label_mode = "binary",
    image_size=(64,64),
    batch_size=100,
)



model = keras.Sequential()


model.add(keras.layers.Conv2D(32,(3,3),padding = "same",input_shape=(64,64,3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.ReLU())
model.add(keras.layers.MaxPool2D((2,2)))

model.add(keras.layers.Conv2D(64,(5,5),padding = "same"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.ReLU())

model.add(keras.layers.MaxPool2D((2,2)))



model.add(keras.layers.Conv2D(256,(3,3),padding = "same"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.ReLU())
model.add(keras.layers.MaxPool2D((2,2)))

model.add(keras.layers.Dropout(0.25))




model.add(keras.layers.Flatten())


model.add(keras.layers.Dense(128))
model.add(keras.layers.Dropout(0.50))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.ReLU())
model.add(keras.layers.Dense(64))
# model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit_generator(train_dataset,
        #  steps_per_epoch = 150,
         epochs = 70,
         validation_data = test_dataset,
         verbose = 2      
         )




# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(2)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()

# batch_holder = np.zeros((20,64,64,3))
img_dir = 'predict/unknown'
ans = []
for i,img in enumerate(os.listdir(img_dir)):
  
  img = tf.keras.preprocessing.image.load_img(os.path.join(img_dir,img),target_size = (64,64))
  # print(img)
  image = tf.keras.preprocessing.image.img_to_array(img)
  image = np.expand_dims(image,axis = 0)
  prediction = model.predict(image)
  if prediction == 1:
    ans.append(1)
  else :
    ans.append(0)

save_predictions('0716089',ans)