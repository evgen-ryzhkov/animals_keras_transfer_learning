from scripts.data import AnimalData
from scripts.model import AnimalClassifier


#  --------------
# Step 1
# preparing train, valid and test images
data_o = AnimalData()
# data_o.prepare_train_valid_images()
# data_o.prepare_test_images()
test_files_path = data_o.get_test_files_path()

#  --------------
# Step 2
# building model
clf_o = AnimalClassifier()
# built_model = clf_o.build_model()
# trained_model = clf_o.train_model(built_model)
# clf_o.save_model(trained_model)


#  --------------
# Step 3
# recognize images
loaded_model = clf_o.load_my_model()
clf_o.recognize_animals(loaded_model, arr_images_path=test_files_path)
