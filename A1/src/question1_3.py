import sys
import os
import cv2
import numpy as np
import pickle
import json
import numdifftools as nd
from scipy import ndimage as ndi
from itertools import combinations_with_replacement
from surf_utils import peak_local_max
import time

IMAGES_DIR = './images/'
TRAIN_DIR = './train/'
JSON_SAVE = 'log_blobs_'
LEVEL = 5
THRESHOLD = 0.08
INIT_SIGMA = 1.35
SIGMA_SCALE = 1.24
EPS = 2.220446049250313e-16
COUNTER = 0

def read_images():
	image_files = os.listdir(IMAGES_DIR)
	images = []

	for i, file in enumerate(image_files):
		img = cv2.imread(IMAGES_DIR+file)
		images.append(img)
		if(i%100 == 0):
			print(i)

	return images

def read_image(img_index):
	image_files = os.listdir(IMAGES_DIR)
	images = []

	for i, file in enumerate(image_files):
		if(img_index == i):
			img = cv2.imread(IMAGES_DIR+file)
			return img, file
	return None, file

def integral_image(image):
# referred from official scikit-image codebase: https://github.com/scikit-image/scikit-image/blob/master/skimage/transform/integral.py
	S = image
	for i in range(image.ndim):
		S = S.cumsum(axis=i)
	return S

def return_sigma(image):
	height, width = image.shape[0], image.shape[1]
	sigma = np.zeros(LEVEL+1)

	sigma[0] = INIT_SIGMA

	for i in range(LEVEL):
		sigma[i+1] = sigma[i]*SIGMA_SCALE
	
	return sigma

def compute_partial_second_order_grads(image, sigma):
# referred from official scikit-image codebase: https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/corner.py
	gaussian_filtered = ndi.gaussian_filter(image, sigma=sigma)

	gradients = np.gradient(gaussian_filtered)
	axes = range(image.ndim)
	axes = reversed(axes)

	H_elems = [np.gradient(gradients[ax0], axis=ax1)
			   for ax0, ax1 in combinations_with_replacement(axes, 2)]

	return H_elems

def surf_feature(img):    
	global COUNTER
	dim = (100, 100)
	resized_image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	gray_image = cv2.normalize(cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY).astype('float'), None, 0, 1, cv2.NORM_MINMAX)

	integral_img = integral_image(gray_image)
	sigma = return_sigma(gray_image)

	all_blobs = []

	for i in range(len(sigma)):
		curr_sigma = sigma[i]
		x, y, z = compute_partial_second_order_grads(integral_img, curr_sigma)
		det_h = np.multiply(x, z) - 0.81 * (y ** 2)
		blobs = peak_local_max(det_h)

		for blob in blobs:
			all_blobs.append((blob[0], blob[1], curr_sigma))

	no_of_blobs=len(all_blobs)
	for ind, center in enumerate(all_blobs):
		radius = int(np.ceil(np.sqrt(2)*sigma[int(center[2])]))
		cv2.circle(resized_image,(center[1],center[0]),radius, (0,0,255))

	if not os.path.exists('surf_blobs'):
		os.makedirs('surf_blobs')

	cv2.imwrite('./surf_blobs/'+str(COUNTER)+'.png', resized_image)
	COUNTER += 1  
	return all_blobs

def merge_jsons():
	dic = json.load(open(JSON_SAVE+'1.json'))

	for i in range(2, 6):
		x = json.load(open(JSON_SAVE+str(i)+'.json'))
		dic.update(x)

	return dic

if __name__ == '__main__':
	image_files = os.listdir(IMAGES_DIR)

	all_image_blobs = {}

	# for i in range(3): 
	# for i in range(0, 100):
	start_time = time.time()
	for i in range(len(image_files)):
		img, file = read_image(i)
		res = surf_feature(img)
		print(i)
	end_time = time.time()
	print('Total time: '+str(end_time-start_time))

		# all_image_blobs[file]=res

	# print(all_image_blobs.items())
	# with open('final_surf_blobs.json', 'wb') as json_file:
		# pickle.dump(all_image_blobs, json_file)