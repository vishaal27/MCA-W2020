import os
import numpy as np
import matplotlib.pyplot as plt
import wave
import pylab
from question1 import *
import pickle
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.externals import joblib

training_path = './training/'
val_path = './validation'
noise_path = './_background_noise_'
training_dict = {}
val_dict = {}

labels_encoder = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}

def augmented_signal(signal, noise_factor):
	noise_files = os.listdir(noise_path)
	file = np.random.choice(noise_files)
	
	noise_signal, frame_rate = read_file(noise_path+'/'+file)
	
	index = np.random.randint(0, len(noise_signal)-len(signal)-10)
	noise = noise_signal[index:index+len(signal)]

	aug_signal = signal + noise_factor*noise
	return aug_signal

def get_label(key):
	global labels_encoder

	x1 = key.find('/')
	x2 = key.find('/', x1+1)
	x3 = key.find('/', x2+1)

	lab = labels_encoder[key[x2+1:x3]]
	return int(lab)

if __name__ == '__main__':
	sampling_rate = 16000

	# FEATURE EXTRACTION

	# training_spec_feats = []
	# val_spec_feats = []

	# for i, (path, subdirs, files) in enumerate(os.walk(training_path)):
	# 	for j, name in enumerate(files):
	# 		training_dict[os.path.join(path, name)] = 1

	# for i, (path, subdirs, files) in enumerate(os.walk(val_path)):
	# 	for j, name in enumerate(files):
	# 		val_dict[os.path.join(path, name)] = 1
			
	# print('training files: '+str(len(training_dict)))
	# print('val files: '+str(len(val_dict)))

	# for i, key in enumerate(tqdm(training_dict)):	
	# 	wav_file = key
	# 	file_series, frame_rate = read_file(wav_file)
	# 	file_series = list(np.pad(np.asarray(file_series), (0, sampling_rate-len(file_series)), mode='constant', constant_values=0))
		
	# 	aug_signal = augmented_signal(file_series, 0.005)
	# 	log_spectrogram = spectrogram(file_series, sampling_rate, 128, 0)
	# 	aug_log_spectrogram = spectrogram(aug_signal, sampling_rate, 128, 0)

	# 	print(key, log_spectrogram.shape)
	# 	print(key, aug_log_spectrogram.shape)
	# 	lab = get_label(key)
	# 	training_spec_feats.append((log_spectrogram, lab))
	# 	training_spec_feats.append((aug_log_spectrogram, lab))

	# pickle.dump(training_spec_feats, open('./augmented_train_spectrogram_features.pickle', 'wb'))

	# for i, key in enumerate(tqdm(val_dict)):	
	# 	wav_file = key
	# 	file_series, frame_rate = read_file(wav_file)
	# 	file_series = list(np.pad(np.asarray(file_series), (0, sampling_rate-len(file_series)), mode='constant', constant_values=0))
		
	# 	log_spectrogram = spectrogram(file_series, sampling_rate, 128, 0)

	# 	print(key, log_spectrogram.shape)
	# 	lab = get_label(key)
	# 	val_spec_feats.append((log_spectrogram, lab))

	# pickle.dump(val_spec_feats, open('./augmented_val_spectrogram_features.pickle', 'wb'))

	# SVM TRAINING AND EVALUATION

	training_data = np.asarray(pickle.load(open('./augmented_train_spectrogram_features.pickle', 'rb')))
	training_x, training_y = training_data[:, 0], training_data[:, 1]

	val_data = np.asarray(pickle.load(open('./original_val_spectrogram_features.pickle', 'rb')))
	val_x, val_y = val_data[:, 0], val_data[:, 1]

	train_x = np.zeros((training_x.shape[0], training_x[0].shape[0], training_x[0].shape[1]))
	validation_x = np.zeros((val_x.shape[0], val_x[0].shape[0], val_x[0].shape[1]))

	for i, _ in enumerate(training_x):
		train_x[i] = training_x[i]
		if(i<len(val_x)):
			validation_x[i] = val_x[i]

	train_y, validation_y = list(np.array(training_y, dtype=np.uint8)), list(np.array(val_y, dtype=np.uint8))
	train_x, validation_x = train_x.reshape((train_x.shape[0], train_x.shape[1]*train_x.shape[2])), validation_x.reshape((validation_x.shape[0], validation_x.shape[1]*validation_x.shape[2]))

	train_x = np.where(np.isinf(train_x), np.nan, train_x)
	validation_x = np.where(np.isinf(validation_x), np.nan, validation_x)

	train_x = SimpleImputer().fit_transform(train_x)
	validation_x = SimpleImputer().fit_transform(validation_x)

	train_x = preprocessing.scale(train_x)
	validation_x = preprocessing.scale(validation_x)

	print('Training X shape: '+str(train_x.shape))
	print('Validation X shape: '+str(validation_x.shape))

	C_vals = [0.01, 0.2, 0.5, 1, 2, 5]
	kernels = ['linear', 'poly', 'rbf', 'sigmoid']

	f = open('./results_spectrogram_augmented.txt', 'a+')

	for c in C_vals:
		for ker in kernels:
			print('---------------------------------------')
			f.write('---------------------------------------\n')
			svm = SVC(gamma='auto', C=c, kernel=ker)
			print('Training SVM ('+str(c)+', '+str(ker)+')')
			f.write('Training SVM ('+str(c)+', '+str(ker)+')\n')
			svm.fit(train_x, train_y)
			print('Evaluating SVM')
			
			train_acc = accuracy_score(train_y, svm.predict(train_x))
			val_acc = accuracy_score(validation_y, svm.predict(validation_x))
			
			joblib.dump(svm, './svm_augmented_spectrogram_('+str(c)+', '+str(ker)+').joblib')

			print('Training accuracy: '+str(train_acc))
			print('Validation accuracy: '+str(val_acc))
			f.write('Training accuracy: '+str(train_acc)+'\n')
			f.write('Validation accuracy: '+str(val_acc)+'\n')
			f.write('---------------------------------------\n')

	f.close()