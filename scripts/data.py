import settings
import glob
import os
import shutil
from PIL import Image
from resizeimage import resizeimage
from random import shuffle


class AnimalData:

	def prepare_train_valid_images(self):

		print('Preparing images...')
		self._clear_directory(settings.PREPARED_TRAIN_IMG_DIR)
		self._clear_directory(settings.PREPARED_VALID_IMG_DIR)

		category_dirs = self._create_category_directories(original_img_dir_path=settings.ORIGINAL_IMG_DIR,
														   prepared_img_dir_path=settings.PREPARED_TRAIN_IMG_DIR)

		category_dirs = self._create_category_directories(original_img_dir_path=settings.ORIGINAL_IMG_DIR,
														  prepared_img_dir_path=settings.PREPARED_VALID_IMG_DIR)

		validation_dataset_size = 0.2
		total_train_files = 0
		total_valid_files = 0

		for dir in category_dirs:
			filenames = glob.glob(settings.ORIGINAL_IMG_DIR + dir + '/' + '*.jpg')

			# random mixing files
			shuffle(filenames)

			# dividing original dataset on train and valid parts
			train_dataset_size = int(len(filenames) * (1 - validation_dataset_size))
			valid_dataset_size = int(len(filenames) * validation_dataset_size)

			train_dataset = filenames[0:train_dataset_size]
			valid_dataset = filenames[-valid_dataset_size:]

			total_train_files += train_dataset_size
			total_valid_files += valid_dataset_size

			train_path = settings.PREPARED_TRAIN_IMG_DIR + dir + '/'
			self._prepare_images_for_dataset(train_dataset, train_path)
			valid_path = settings.PREPARED_VALID_IMG_DIR + dir + '/'
			self._prepare_images_for_dataset(valid_dataset, valid_path)

		print('Test and valid image were prepared successfully.')
		print('Total train images = ', total_train_files)
		print('Total valid images = ', total_valid_files)

	def prepare_test_images(self):
		self._clear_directory(settings.PREPARED_TEST_IMG_DIR)
		original_files = glob.glob(settings.ORIGINAL_TEST_IMG_DIR + '*.jpg')
		test_path = settings.PREPARED_TEST_IMG_DIR
		self._prepare_images_for_dataset(original_files, test_path)
		print('Test images ({}) were successfully prepared.'.format(len(original_files)))

	@staticmethod
	def get_test_files_path():
		return glob.glob(settings.PREPARED_TEST_IMG_DIR + '*.jpg')

	@staticmethod
	def _clear_directory(path):
		for root, dirs, files in os.walk(path):
			for f in files:
				os.unlink(os.path.join(root, f))
			for d in dirs:
				shutil.rmtree(os.path.join(root, d))

	@staticmethod
	def _create_category_directories(original_img_dir_path, prepared_img_dir_path):
		category_dirs = os.listdir(original_img_dir_path)
		for dir in category_dirs:
			category_dir = prepared_img_dir_path + '/' + dir

			if not os.path.exists(category_dir):
				os.makedirs(category_dir)
		return category_dirs

	@staticmethod
	def _prepare_images_for_dataset(dataset, path_where_to_save):
		# resize images to inception_v3 format
		img_width = settings.INPUT_IMG_SIZE
		img_height = settings.INPUT_IMG_SIZE
		file_name_extension = '.jpg'

		for idx, img in enumerate(dataset):
			fd_img = open(img, 'rb')
			img_r = Image.open(fd_img)

			# resize with saving aspect ratio
			preprocessed_image = resizeimage.resize_contain(img_r, [img_width, img_height])
			preprocessed_image.save(path_where_to_save + str(idx) + file_name_extension,
									preprocessed_image.format)
			fd_img.close()

