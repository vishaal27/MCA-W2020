import sys
import os
import cv2
import numpy as np
import pickle
import json
import time
from copy import deepcopy

IMAGES_DIR = './images/'
TRAIN_DIR = './train/'
JSON_SAVE = 'log_blobs_'
LEVEL = 5
THRESHOLD = 0.03
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

def output_image_size(image, filter_size, padding, stride):
	size_x, size_y = image.shape[0], image.shape[1]

	x_dim = (size_x - filter_size + 2*padding)//stride
	x_dim += 1

	y_dim = (size_y - filter_size + 2*padding)//stride
	y_dim += 1
	
	out_size = np.zeros((x_dim, y_dim), dtype=float)
	return out_size

def convolve(image, kernel):
	filter_size = len(kernel)
	height, width = image.shape[0], image.shape[1]
	padding = (filter_size - 1)//2
	padded_image = np.pad(image, padding)
	stride = 1
	output_image = output_image_size(image, filter_size, padding, stride)
	
	for col in range(padding, height + padding):
			for row in range(padding, width + padding):
				receptive_field = padded_image[col-padding: col+padding+1, row-padding: row+padding+1]
				response = (receptive_field*kernel).sum()
				output_image[col-padding, row-padding] = response
				
	return output_image

def log_filter(sigma):
	kernel_size = np.round(6*sigma)
	if(kernel_size%2==0):
		kernel_size += 1

	range_arr = np.arange(-1*(kernel_size//2), (kernel_size//2)+1)	
	x, y = np.meshgrid(range_arr, range_arr)
	
	exponent_num = -1*(np.square(x)+np.square(y))
	exponent_denom = 2*(sigma**2)
	exponential = np.exp( exponent_num / exponent_denom )

	max_exp = exponential.max()
	threshold =  max_exp*EPS
	exponential[exponential<threshold] = 0

	if(exponential.sum() == 0):
		pass
	else:
		exponential = exponential/exponential.sum()

	kernel_denom = sigma**2
	kernel_num = np.square(x) + np.square(y) - 2*kernel_denom
	
	kernel_out = -1*exponential*kernel_num/kernel_denom
	kernel_out -= kernel_out.mean()

	return kernel_out

def check_valid(i, j, n_x, n_y, size_x, size_y):
	return 0<=(i+n_x)<size_x and 0<=(j+n_y)<size_y

def NMS_validation_condition(query_ind, sig_index, i, j, scale_space, size_x, size_y):
	bool_list = []
	neighbours = [[1, 0], [-1, 0], [0, 1], [0, -1], [-1, -1], [-1, 1], [1, -1], [1, 1]]
	
	for neighbour in neighbours:
		n_x, n_y = neighbour[0], neighbour[1]
		if(check_valid(i, j, n_x, n_y, size_x, size_y)):
			bool_list.append(scale_space[i][j][sig_index]>scale_space[i+n_x][j+n_y][query_ind])

	if(False in bool_list):
		return False
	else:
		return True

def NMS(sigma, scale_space):
	temp_scale_space = deepcopy(scale_space)
	image_mask = np.zeros(LEVEL)

	for i in range(LEVEL):
		sigma_coeff = np.sqrt(2)*sigma[i]
		radius = int(np.ceil(sigma_coeff))
		image_mask[i] = radius
	
	size_x, size_y = scale_space[..., 0].shape[0], scale_space[..., 1].shape[1]
	
	blob_triples = []

	for sig_index in range(LEVEL):
		temp_scale_space[: int(image_mask[sig_index]), :int(image_mask[sig_index]), sig_index] = 0
		temp_scale_space[-1*int(image_mask[sig_index]): , -1*int(image_mask[sig_index]): , sig_index] = 0
		
		range_x = (int(image_mask[sig_index])+1, size_x-1-int(image_mask[sig_index]))
		range_y = (int(image_mask[sig_index])+1, size_y-1-int(image_mask[sig_index]))

		for i in range(range_x[0], range_x[1]):
			for j in range(range_y[0], range_y[1]):
				
				if(scale_space[i, j, sig_index]>=THRESHOLD):
					pass
				else:
					continue

				curr_condition_res = NMS_validation_condition(sig_index, sig_index, i, j, scale_space, size_x, size_y)
				
				bottom_condition_res = True
				top_condition_res = True
				
				if(sig_index>=1):
					bottom_condition_res = NMS_validation_condition(sig_index-1, sig_index, i, j, scale_space, size_x, size_y)
					bottom_condition_res = bottom_condition_res and scale_space[i, j, sig_index]>scale_space[i, j, sig_index-1]

				if(sig_index<LEVEL-1):
					top_condition_res = NMS_validation_condition(sig_index+1, sig_index, i, j, scale_space, size_x, size_y)
					top_condition_res = top_condition_res and scale_space[i, j, sig_index]>scale_space[i, j, sig_index + 1] 					
					
				if curr_condition_res and bottom_condition_res and top_condition_res:
					temp_scale_space[i][j][sig_index] = 1
					blob_triples.append([i, j, sig_index])  

	return blob_triples

def return_scale_and_space(image):
	height, width = image.shape[0], image.shape[1]
	scale_space = np.zeros((height, width, LEVEL))
	scale_space = scale_space.astype(np.float32)
	sigma = np.zeros(LEVEL+1)

	sigma[0] = INIT_SIGMA

	for i in range(LEVEL):
		kernel = log_filter(sigma[i])
		convolved_image = convolve(image, kernel)
		scale_space[...,i] = np.square(convolved_image)
		sigma[i+1] = sigma[i]*SIGMA_SCALE
	
	return sigma, scale_space

  
def log_feature(img):  
	global COUNTER  
	dim = (100, 100)
	resized_image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	gray_image = cv2.normalize(cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY).astype('float'), None, 0, 1, cv2.NORM_MINMAX)

	sigma, scale_space = return_scale_and_space(gray_image) 
	all_blobs = NMS(sigma, scale_space) 

	no_of_blobs=len(all_blobs)
	for ind, center in enumerate(all_blobs):
		radius = int(np.ceil(np.sqrt(2)*sigma[center[2]]))
		cv2.circle(resized_image,(center[1],center[0]),radius, (0,0,255))

	if not os.path.exists('log_blobs'):
		os.makedirs('log_blobs')

	cv2.imwrite('./log_blobs/'+str(COUNTER)+'.png', resized_image)
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

	start_time = time.time()

	for i in range(len(image_files)): 
	# for i in range(0, 100):
		img, file = read_image(i)
		res = log_feature(img)
		print(i)
		# all_image_blobs[file]=res
	end_time = time.time()
	print('Total time: '+str(end_time-start_time))
	# print(all_image_blobs)
	# with open('log_blobs_1.json', 'w') as json_file:
	# 	json.dump(all_image_blobs, json_file)

	# final_merged_dict = merge_jsons()
	# with open('final_log_blobs.json', 'w') as json_file:
		# json.dump(final_merged_dict, json_file)