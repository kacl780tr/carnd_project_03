import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn import model_selection
from skimage import io as skio
from skimage import transform as sktm
import os
from pathlib import Path

def rescale_channel(batch):
	"""
	Rescale all channels of an image to (-1, 1) independently
	param: batch: a 4D array of immages with shape [index, height, width, depth]
	return: rescaled images with data type np.float32
	"""
	batch = batch.astype(np.float32)
	min = np.min(batch, axis=(1,2), keepdims=True)		# not sure this is a good idea, might cause color balance issues
	max = np.max(batch, axis=(1,2), keepdims=True)
	mu = 0.5*(max + min)
	rg = 0.5*(max - min + 1.0e-6)
	batch = (batch - mu)/rg
	return batch


def rescale_flat(batch):
	"""
	Rescale all channels of an image to (-1, 1) simultaneously
	param: batch: a 4D array of immages with shape [index, height, width, depth]
	return: rescaled images with data type np.float32
	"""
	batch = batch.astype(np.float32)
	min = np.min(batch, axis=(1,2,3), keepdims=True)
	max = np.max(batch, axis=(1,2,3), keepdims=True)
	mu = 0.5*(max + min)
	rg = 0.5*(max - min + 1.0e-6)
	batch = (batch - mu)/rg
	return batch


def rescale_pass(batch):
	"""
	Apply no rescaling
	param: batch: a 4D array of images with shape [index, height, width, depth]
	return: images with data type np.float32
	"""
	return batch.astype(np.float32)


class Data(object):
	"""
	A class for self-contained handling of datasets
	"""
	def __init__(self, feature, label, scaler=None):
		"""
		Initialize
		param: feature: data features, here assumed to be 4d numpy array
		param: label: data labels, assumed to be 1d numpy array
		param: scaler: the scaling function to be applied to generated batches (default = rescale_channel)
		"""
		assert len(feature) == len(label)
		self.feature = feature
		self.label = label
		self.length = len(self.feature)
		self.__data = None
		self.__scaler = scaler
		if scaler is None:
			self.__scaler = rescale_pass # we use keras facilities for this due to connection with simulator

	def input_shape(self):
		return self.feature[0].shape	# not sure if this will work

	def scale_function(self):
		return self.__scaler

	def shuffle(self):
		self.__data = (self.feature, self.label)
		self.feature, self.label = sklearn.utils.shuffle(self.feature, self.label)

	def reset(self):
		if self.__data:
			self.feature = self.__data[0]
			self.label = self.__data[1]
			self.__data = None

	def batch_count(self, size_batch):
		n = self.length // size_batch
		if self.length % size_batch:
			n += 1
		return n

	def make_batches(self, size_batch):
		for b in range(0, self.length, size_batch):
			x = self.feature[b:b+size_batch]
			y = self.label[b:b+size_batch]
			yield self.__scaler(x), y.astype(np.float32), len(x)

	def make_batch(self):
		return self.__scaler(self.feature), self.label.astype(np.float32), self.length

	def subset(self, n):
		return Data(self.feature[:n], self.label[:n], scaler=self.__scaler)


def load_log(datadir):
	logfile = os.path.join(datadir, "driving_log.csv")
	log_drv = pd.read_csv(logfile, header=0)
	return log_drv


def read_images(pathlist):
	images = np.zeros((len(pathlist), 160, 320, 3), dtype=np.uint8)
	for i, pth in enumerate(pathlist):
		img = skio.imread(pth)
		images[i,:,:,:] = img
	return images


def data_summary(train, valid=None, test=None):
	"""
	Print summary statistics of data sets
	param: train: training data set
	param: valid: validation data set (optional)
	param: test: testing data set (optional)
	return: nothing
	"""
	print("Number of training examples = {}".format(train.length))
	if valid: print("Number of validation examples = {}".format(valid.length))
	if test: print("Number of testing examples = {}".format(test.length))
	print("Image data shape = {}".format(train.input_shape()))


def concatenate_data(datalist):
	data_base = datalist[0]
	for dat in datalist[1:]:
		features = np.concatenate([data_base["features"], dat["features"]])
		steering = np.concatenate([data_base["steering"], dat["steering"]])
		data_base["features"] = features
		data_base["steering"] = steering
	return data_base


def prepare_data(datadir, camera):
	data_log = load_log(datadir)
	angles_steer = data_log["steering"]
	angles_steer = np.array(angles_steer)
	path_camera = data_log[camera]
	images_camera = read_images(path_camera)
	data = {}
	data["features"] = images_camera
	data["steering"] = angles_steer
	return data


def save_data(filename, dict):
	path = Path(filename)
	if path.exists():
		path.unlink()
	with open(filename, mode="wb") as f:
		pickle.dump(dict, f)


def archive_data(source, target, camera):
	data = prepare_data(source, camera)
	archive = load_archive(target)
	if archive:
		archive = concatenate_data([archive, data])
	else:
		archive = data;
	save_data(target, archive)


def load_archive(filename):
	path = Path(filename)
	if path.exists():
		with open(filename, mode="rb") as f:
			data_archive = pickle.load(f)
		return data_archive
	else:
		return None


def augment_data(source):
	"""
	Generate left-right flipped dataset (as dictionary) from source dataset
	param: source: base dataset
	return: flipped dataset dictionary
	"""
	features = source["features"]
	steering = source["steering"]
	flipped = np.zeros_like(features)
	for i in range(features.shape[0]):
		img = features[i,:,:,:]
		img = np.fliplr(img)
		flipped[i,:,:,:] = img
	data_flipped = {}
	data_flipped["features"] = flipped
	data_flipped["steering"] = -1.0*steering
	return data_flipped


def load_dataset(filenames, lrbias=0.10, frac_test=0.0, scaler=None):
	"""
	Load data from pickle archives and create one or optionally two Data objects
	param: filenames: a list of filenames from which to read data
	param: lrbias: if the filename contains "left" or "right" indicating off-center camera
					then add the lrbias to the left steering angles and subtract same from right
	param: frac_test: if > 0 then split off that fraction as a test set
	param: scaler: an optional scaling function to be applied when batches are generated
	"""
	archives = []
	for fn in filenames:
		archive = load_archive(fn)
		if archive is None:
			continue
		bias = 0.0
		if fn.find("left") >= 0:
			bias = lrbias
		elif fn.find("right") >= 0:
			bias = -lrbias
		if fn.find("-f") >= 0:	# flipped data - reverse bias sign
			bias *= -1.0
		if bias != 0.0:
			archive["steering"] += bias
		archives.append(archive)
	data = concatenate_data(archives)
	if frac_test > 0.0 and frac_test < 1.0:
		X, Xt, y, yt = model_selection.train_test_split(data["features"], data["steering"], test_size=frac_test, random_state=4357)
		return Data(X, y, scaler=scaler), Data(Xt, yt, scaler=scaler)
	return Data(data["features"], data["steering"], scaler=scaler)


def make_archive_from_dump():
	path = "./data/camera_"
	camera = ["center", "left", "right"]
	for cm in camera:
		path_source = path + cm + ".p"
		path_target = path + cm + "-f.p"
		data_base = load_archive(path_source)
		if data_base:
			data_flip = augment_data(data_base)
			save_data(path_target, data_flip)


if __name__ == "__main__":
	target = "./data/track_2_"
	camera = ["center", "left", "right"]
	for cm in camera:
		sn = target + cm + ".p"
		tn = target + cm + "-f.p"
		base = load_archive(sn)
		augm = augment_data(base)
		save_data(tn, augm)
