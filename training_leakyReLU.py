from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras.layers import LeakyReLU
from keras import backend as K 
import matplotlib.pyplot as plt
  
img_width, img_height = 224, 224

train_data_dir = 'Dataset/train'
validation_data_dir = 'Dataset/test'
nb_train_samples = 8192
nb_validation_samples = 2048
epochs = 16
batch_size = 32

if K.image_data_format() == 'channels_first': 
	input_shape = (3, img_width, img_height) 
else: 
	input_shape = (img_width, img_height, 3) 

model = Sequential() 
model.add(Conv2D(32, (2, 2), input_shape=input_shape)) 
model.add(LeakyReLU(alpha=0.1)) 
model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Conv2D(32, (2, 2))) 
model.add(LeakyReLU(alpha=0.1)) 
model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Conv2D(64, (2, 2))) 
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Flatten()) 
model.add(Dense(64)) 
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5)) 
model.add(Dense(1)) 
model.add(Activation('sigmoid')) 


model.compile(loss='binary_crossentropy', 
			optimizer='rmsprop', 
			metrics=['accuracy']) 

train_datagen = ImageDataGenerator( 
	rescale=1. / 255, 
	shear_range=0.2, 
	zoom_range=0.2, 
	horizontal_flip=True) 

test_datagen = ImageDataGenerator(rescale=1. / 255) 

train_generator = train_datagen.flow_from_directory( 
	train_data_dir, 
	target_size=(img_width, img_height), 
	batch_size=batch_size, 
	class_mode='binary') 

validation_generator = test_datagen.flow_from_directory( 
	validation_data_dir, 
	target_size=(img_width, img_height), 
	batch_size=batch_size, 
	class_mode='binary') 

history = model.fit( 
	train_generator, 
	steps_per_epoch=nb_train_samples // batch_size, 
	epochs=epochs, 
	validation_data=validation_generator, 
	validation_steps=nb_validation_samples // batch_size) 

model.save('my_model.h5') 

loss_train = history.history['accuracy']
loss_val = history.history['val_accuracy']
epochs = range(1, 17)
plt.plot(epochs, loss_train, 'g', label='Training accuracy')
plt.plot(epochs, loss_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy (leakyReLU)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# single image prediction :

import numpy as np 
from keras.preprocessing import image

test_image = image.load_img('single_prediction/Positive.png', target_size = (224, 224))
test_image = image.img_to_array(test_image) # an array of 2d to 3d (rgb)
# expanding the test_img dimensions to make it compatible (4d) for the predict method :
test_image = np.expand_dims(test_image, axis = 0)

result = model.predict(test_image)

if result[0][0] == 1 :
    prediction = 'Covid Positive !!'
else :
    prediction = 'Covid Negative :)'

print(prediction)
