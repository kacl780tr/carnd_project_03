import drive_data
import train_tools as tt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.layers import Cropping2D, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def make_callbacks(dump_path):
	"""
	Create callback objects for managing the training process
	param: dump_path: the file path to which the model checkpoints will be saved
	return: a list of checkpoint objects
	"""
	checkpoint = ModelCheckpoint(dump_path, verbose=1, save_best_only=True, mode="min")
	rate_decay = ReduceLROnPlateau(factor=0.75, patience=3, verbose=1, min_lr=0.00005)
	terminator = EarlyStopping(patience=10, verbose=1, mode="min")
	return [checkpoint, rate_decay, terminator]


def make_modelinput(data, crop=True):
	"""
	Create the input layers of the network, specifically the cropping and normalization layers
	param: data: the training data set to be used
	param: crop: optional switch to select cropping (defaults to true)
	return: a sequential model with input layers
	"""
	model = Sequential()
	if crop:
		model.add(Cropping2D(cropping=((55, 25), (0,0)), input_shape=data.input_shape()))
		model.add(Lambda(lambda x: ((x/255.0) - 0.5)))
	else:
		model.add(Lambda(lambda x: ((x/255.0) - 0.5), input_shape=data_trn.input_shape())) 
	return model


def make_nvidianet(data, crop=True, rate_drop=0.0):
	"""
	Create the network architecture - here basically the Nvidia network as described in project description
	param: data: the training data to be used
	param: crop: optional switch to select input cropping (defaults to true)
	param: rate_drop: the drop rate for dropout regularization (default = 0) if > 0 then network architecture is
						altered slightly to prevent narrowing of feature maps in final convolution layers
	return: fully specified network architecture
	"""
	model = make_modelinput(data, crop=crop)
	model.add(Conv2D(24, 5, 5, subsample=(2,2), activation="relu"))
	model.add(Conv2D(36, 5, 5, subsample=(2,2), activation="relu"))
	model.add(Conv2D(48, 5, 5, subsample=(2,2), activation="relu"))
	if rate_drop > 0.0:
		model.add(Conv2D(64, 3, 3, activation="relu", border_mode="same"))	# prevent further x-y shrinking of feature maps
		model.add(Conv2D(64, 3, 3, activation="relu", border_mode="same"))
	else:
		model.add(Conv2D(64, 3, 3, activation="relu"))
		model.add(Conv2D(64, 3, 3, activation="relu"))
	model.add(Flatten())
	if rate_drop > 0.0: 
		model.add(Dropout(rate_drop))
	model.add(Dense(100, activation="relu"))
	model.add(Dense(50, activation="relu"))
	model.add(Dense(1))
	return model


def train_model(model, data_train, data_test=None, dump_path="./models/unknown.h5"):
	"""
	Train the provided model on the provided training data
	param: model: the model to be trained
	param: data_train: the training data
	param: data_test: the testing data (optional)
	param: dump_path: the path to which checkpoint files are to be saved
	return: the training history object
	"""
	trainer = Adam()
	model.compile(optimizer=trainer, loss="mse")	# use mean-squared error
	X_trn, y_trn, _ = data_train.make_batch()		# convert uint8 arrays to float32 for training
	callbacks = make_callbacks(dump_path)			# callbacks for learning rate decay, checkpoint save, and early stopping
	train_history = model.fit(X_trn, y_trn, batch_size=32, nb_epoch=100, validation_split=0.25, shuffle=True, callbacks=callbacks)
	X_trn, y_trn = None, None						# attempt to recovery memory
	if data_test:
		X_tst, y_tst, _ = data_test.make_batch()
		test_loss = model.evaluate(X_tst, y_tst)
		print("Test loss for {} = {:>3.5f}".format(dump_path, test_loss))
	return train_history


def get_datafiles():
	"""
	Create and return list of data files for training
	"""
	data_files = [
		"./data/track_1_center.p", 
		"./data/track_1_center-f.p",
		"./data/track_1_left.p",
		"./data/track_1_left-f.p",
		"./data/track_1_right.p",
		"./data/track_1_right-f.p",
		"./data/track_2_center.p", 
		"./data/track_2_center-f.p",
		"./data/track_2_left.p",
		"./data/track_2_left-f.p",
		"./data/track_2_right.p",
		"./data/track_2_right-f.p"
		]
	return data_files


def run_training(wide=False):
	"""
	Run the training loop
	param: wide: switch to 
	"""
	data_files = get_datafiles()
	
	data_trn, data_tst = drive_data.load_dataset(data_files, frac_test=0.20)

	model_nvidia = make_nvidianet(data_trn)
	train_model(model_nvidia, data_trn, data_tst, dump_path="model.h5")


def run_data_summary():
	data_files = get_datafiles()
	files_track_1, files_track_2 = [], []
	for fn in data_files:
		if fn.find("track_1") >=0:
			files_track_1.append(fn)
		if fn.find("track_2") >= 0:
			files_track_2.append(fn)

	data_train, data_test = drive_data.load_dataset(files_track_1, frac_test=0.20)
	drive_data.data_summary(data_train, test=data_test)
	data_train, data_test = drive_data.load_dataset(files_track_2, frac_test=0.20)
	drive_data.data_summary(data_train, test=data_test)
	

if __name__ == "__main__":
 	run_training()

