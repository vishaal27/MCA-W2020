import os
import glob
import pickle
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import confusion_matrix

training_path = './training/'
val_path = './validation'

if __name__ == '__main__':
	model_files = glob.glob('./*.joblib')

	training_data = np.asarray(pickle.load(open('./original_train_spectrogram_features.pickle', 'rb')))
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

	for model in model_files:
		model_name = model
		model = joblib.load(model)
		print('---------------------------------------')
		
		try:
			preds = model.predict(validation_x)
			val_acc = accuracy_score(validation_y, preds)
			print(model_name)
			print('Validation accuracy: '+str(val_acc))
			print(confusion_matrix(validation_y, preds))
			# print('Validation score report: ')
			# print(classification_report(validation_y, preds))
			# print('Validation precision')
			# print(precision_score(validation_y, preds))
			# print('Validation recall')
			# print(recall_score(validation_y, preds))
		except ValueError as e:
			continue

		print('---------------------------------------')