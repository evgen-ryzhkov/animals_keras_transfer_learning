import settings

from keras.applications import inception_v3
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, Flatten

from tensorflow.python.keras.callbacks import TensorBoard

from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib.pyplot as plt


class AnimalClassifier:

	@staticmethod
	def build_model():
		base_model = inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
		# base_model = VGG16(include_top=False, weights='imagenet', input_shape=(299, 299, 3))

		# Freeze the layers except the last 2 layers
		# freeze almost all layers because the current task is very similar for base model pre-trained classes
		for layer in base_model.layers[:-2]:
			layer.trainable = False

		# check if right number of layers are freeze
		for layer in base_model.layers:
			print(layer, layer.trainable)

		model = Sequential([
			base_model,
			Flatten(),
			Dense(64, activation='relu', name='dense_1'),
			Dropout(0.5),
			Dense(3, activation='softmax', name='dense_2')
			])
		model.summary()

		return model

	def train_model(self, model):

		# No Data augmentation
		train_datagen = ImageDataGenerator(rescale=1. / 255)
		validation_datagen = ImageDataGenerator(rescale=1. / 255)

		train_batchsize = 100
		val_batchsize = 10

		# Data Generator for Training data
		train_generator = train_datagen.flow_from_directory(
			settings.PREPARED_TRAIN_IMG_DIR,
			target_size=(settings.INPUT_IMG_SIZE, settings.INPUT_IMG_SIZE),
			batch_size=train_batchsize,
			class_mode='categorical')

		# Data Generator for Validation data
		validation_generator = validation_datagen.flow_from_directory(
			settings.PREPARED_VALID_IMG_DIR,
			target_size=(settings.INPUT_IMG_SIZE, settings.INPUT_IMG_SIZE),
			batch_size=val_batchsize,
			class_mode='categorical',
			shuffle=False)

		tensorboard = TensorBoard(log_dir=settings.lOGS_DIR)

		model.compile(optimizer='adam',
					  loss='categorical_crossentropy',
					  metrics=['accuracy'])

		history = model.fit_generator(
					train_generator,
					steps_per_epoch=train_generator.samples/train_generator.batch_size,
					epochs=15,
					callbacks=[tensorboard],
					validation_data=validation_generator,
					validation_steps=validation_generator.samples/validation_generator.batch_size
		)
		print('training finished')
		self._visualize_model_training(history)

		return model

	@staticmethod
	def _visualize_model_training(history):
		print(history.history.keys())
		plt.plot(history.history['acc'], 'b', label='Training acc')
		plt.plot(history.history['val_acc'], 'r', label='Validation acc')
		plt.title('Training and validation accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('Number of epoch')
		plt.legend(loc='lower right')
		plt.figure()

		plt.plot(history.history['loss'], 'b', label='Training loss')
		plt.plot(history.history['val_loss'], 'r', label='Validation loss')
		plt.title('Training and validation loss')
		plt.ylabel('loss')
		plt.xlabel('Number of epoch')
		plt.legend(loc='lower right')
		plt.show()

	@staticmethod
	def save_model(model):
		try:
			save_model(model, settings.MODELS_DIR + settings.MODEL_FILE_NAME)
			# model.save(settings.MODELS_DIR + settings.MODEL_FILE_NAME)
			print('Model was successfully saved.')
		except IOError:
			raise ValueError('Something wrong with file save operation.')
		except ValueError:
			raise ValueError('Something wrong with model.')

	@staticmethod
	def load_my_model():
		try:
			model = load_model(settings.MODELS_DIR + settings.MODEL_FILE_NAME)
			return model
		except IOError:
			raise ValueError('Something wrong with file save operation.')

	def recognize_animals(self, model, arr_images_path):
		for img_path in arr_images_path:
			self.recognize_animal(model, img_path)

	def recognize_animal(self, model, img_path):

		labels = ['cat', 'horse', 'lion']
		x, img = self._preprocess_input_data(img_path)

		# get the predicted probabilities for each class
		prediction = model.predict(x)

		# convert the probabilities to class labels
		label = labels[np.argmax(prediction)]
		prediction_str = np.array2string(prediction, separator=',', precision=3, suppress_small=True)

		plt_label = 'Label = ' + label + ' Prediction = ' + prediction_str
		print(plt_label)

		plt.imshow(img)
		plt.title(label)
		plt.axis('off')
		plt.show()

	@staticmethod
	def _preprocess_input_data(img_path):
		original_img = load_img(img_path, target_size=(settings.INPUT_IMG_SIZE, settings.INPUT_IMG_SIZE))

		# Convert the PIL array (width, height, channel) into numpy array (height, width, channel)
		numpy_image = img_to_array(original_img)

		# reshape data in terms of batchsize (batchsize, height, width, channels)
		image_batch = np.expand_dims(numpy_image, axis=0)

		# prepare the image for the Inception model
		processed_image = inception_v3.preprocess_input(image_batch.copy())

		return processed_image, original_img
