import os
import cv2
import numpy as np
import pickle
import time

IMAGES_DIR = './images/'
TRAIN_DIR = './train/'
PICKLE_SAVE = 'color_autocorrelogram_features_'
IMAGE_TO_INDEX_DICT = {}
INDEX_TO_IMAGE_DICT = {}

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
			return img
	return None

def populate_image_to_index():
	global IMAGE_TO_INDEX_DICT
	global INDEX_TO_IMAGE_DICT

	image_files = os.listdir(IMAGES_DIR)

	for i, file in enumerate(image_files):
		IMAGE_TO_INDEX_DICT[file] = i
		INDEX_TO_IMAGE_DICT[i] = file

def return_quantised_color(color_ind, quantization_scale):
	norm = 256/quantization_scale
	return color_ind//norm

def quantise_image(img):
	height, width, channels = img.shape

	for i in range(height):
		for j in range(width):
			for k in range(channels):
				img[i][j][k] = return_quantised_color(img[i][j][k], 64)

	return img

def assign_labels(img, labels):
	img = img.reshape((-1, 3))
	ret_labels = np.zeros(len(img))
	
	for i in range(len(img)):	
		ret_labels[i] = labels[img[i][0]][img[i][1]][img[i][2]]
	
	return ret_labels

def quantize_colour_space():
	c_array = np.zeros((64, 64, 64, 3))

	for i in range(64):
		for j in range(64):
			for k in range(64):
				c_array[i][j][k][0], c_array[i][j][k][1], c_array[i][j][k][2] = i, j, k 

	Z = c_array.reshape((-1, 3))
	Z = np.float32(Z)

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K = 64
	_, labels, centroids = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
	return labels, centroids

def correlogram(image, centers, distances):
	size_x, size_y, channels = image.shape[0], image.shape[1], image.shape[2]
	histogram = []

	for dist in distances:
		colour_norms = 0
		colours_counts = [0]*len(centers)
 
		for x in range(0, size_x, int(round(size_x/10))):
			for y in range(0, size_y, int(round(size_y/10))):
				curr_pixel = image[x][y]
				neighbour_set = return_valid_neighbours(image, x, y, dist)
				for neighbour in neighbour_set:
					neighbour_pixel = image[neighbour[0]][neighbour[1]]
 
					for m in range(len(centers)):
						if(np.array_equal(centers[m], curr_pixel) and np.array_equal(centers[m], neighbour_pixel)):
							colour_norms += 1
							colours_counts[m] += 1

		for i in range(len(colours_counts)):
			colours_counts[i] = float(colours_counts[i])/colour_norms
		
		histogram.append(colours_counts)

	return histogram

def autoCorrelogram(img, center, label):

	quantised_img = quantise_image(img)
	
	# scale_percent = 75
	# width = int(img.shape[1] * scale_percent / 100)
	# height = int(img.shape[0] * scale_percent / 100)
	# dim = (width, height)
	
	dim = (100, 100)
	resized_image = cv2.resize(quantised_img, dim, interpolation = cv2.INTER_AREA)

	labels = np.uint8(assign_labels(resized_image, label))
	centered_pixels = center[labels.flatten()]
	centered_image = centered_pixels.reshape((resized_image.shape))
	distances = [1, 3, 5, 7] # taken from the paper: Image Indexing Using Color Correlograms, CVPR 1997
	histogram = correlogram(centered_image, center, distances)
	return histogram

def validity_check(size_x, size_y, point):
	x, y = point[0], point[1]

	if(x<=-1 or x>=size_x):
		return False

	if(y<=-1 or y>=size_y):
		return False

	return True
 
def return_valid_neighbours(image, x, y, distance):
	size_x, size_y = image.shape[0], image.shape[1]

	neighbours = [[1, 0], [-1, 0], [0, 1], [0, -1], [-1, -1], [-1, 1], [1, -1], [1, 1]]

	points = []

	for neighbour in neighbours:
		points.append((x+distance*neighbour[0], y+distance*neighbour[1]))

	valid_neighbours = []
	
	for pt in points:
		if(validity_check(size_x, size_y, pt)==True):
		  valid_neighbours.append(pt)
	return valid_neighbours

def histogram_match_score(hist1, hist2):

	m, k = len(hist1), len(hist1[0])

	s = 0

	for i in range(m):
		for j in range(k):
			s += np.abs(hist1[i][j] - hist2[i][j])/(1 + hist1[i][j] + hist2[i][j])

	return (1/m)*s

def merge_pickle_files():
	ret_list = []

	for i in range(1, 6):
		x = pickle.load(open(PICKLE_SAVE+str(i)+'.pickle', 'rb'))
		ret_list += x
	return ret_list

def return_query_list():
	query_files = os.listdir(TRAIN_DIR+'query')
	
	queries = []

	for file in query_files:
		f=open(TRAIN_DIR+'query/'+file, 'r')
		query_string = f.read()
		query = query_string[5:query_string.find(' ')]
		queries.append(query+'.jpg')

	return queries, query_files

def return_ground_truths(query_file):
	global IMAGE_TO_INDEX_DICT
	query_file = query_file[:-9]

	ground_truths = []

	for file_type in ['good', 'ok', 'junk']:
		f=open(TRAIN_DIR+'ground_truth/'+query_file+file_type+'.txt', 'r')
		for line in f.readlines():
			ground_truths.append(IMAGE_TO_INDEX_DICT[line.strip('\n')+'.jpg']) 

	return ground_truths

def intersection(l1, l2):  
	l2 = set(l2)
	l3 = [value for value in l1 if value in l2] 
	return len(l3)

def return_retrieval_percentages(matching_results, query_file):
	global IMAGE_TO_INDEX_DICT
	query_file = query_file[:-9]

	ground_truths_good = []
	ground_truths_ok = []
	ground_truths_junk = []

	f=open(TRAIN_DIR+'ground_truth/'+query_file+'good'+'.txt', 'r')
	for line in f.readlines():
		ground_truths_good.append(IMAGE_TO_INDEX_DICT[line.strip('\n')+'.jpg']) 

	f=open(TRAIN_DIR+'ground_truth/'+query_file+'ok'+'.txt', 'r')
	for line in f.readlines():
		ground_truths_ok.append(IMAGE_TO_INDEX_DICT[line.strip('\n')+'.jpg']) 

	f=open(TRAIN_DIR+'ground_truth/'+query_file+'junk'+'.txt', 'r')
	for line in f.readlines():
		ground_truths_junk.append(IMAGE_TO_INDEX_DICT[line.strip('\n')+'.jpg']) 

	predictions = [match[1] for match in matching_results]

	percent_good = intersection(predictions, ground_truths_good)/len(ground_truths_good)
	percent_ok = intersection(predictions, ground_truths_ok)/len(ground_truths_ok)
	percent_junk = intersection(predictions, ground_truths_junk)/len(ground_truths_junk)	

	return percent_good, percent_ok, percent_junk

def matching(query_feature, query_img_index, all_image_features, thresholds, ground_truths):
	global INDEX_TO_IMAGE_DICT
	scores = []

	for i, image_idx in enumerate(IMAGE_TO_INDEX_DICT.values()):
		if(image_idx == query_img_index):
			continue
		score = histogram_match_score(query_feature, all_image_features[image_idx])
		scores.append((score, image_idx))

	sorted_scores = sorted(scores, reverse=True)

	average_precisions = []
	average_recalls = []

	precisions_at_k = {}
	recalls_at_k = {}

	for threshold in thresholds:
		top_matches = sorted_scores[:threshold]
		print('At threshold '+str(threshold))
		predictions = [match[1] for match in top_matches]

		intersecting_elements = intersection(predictions, ground_truths)
		curr_precision = intersecting_elements/threshold	
		curr_recall = intersecting_elements/len(ground_truths)	
		
		average_precisions.append(curr_precision)
		average_recalls.append(curr_recall)
		
		precisions_at_k[threshold] = curr_precision
		recalls_at_k[threshold] = curr_recall

		print('Precision@'+str(threshold)+': '+str(curr_precision))
		print('Recall@'+str(threshold)+': '+str(curr_recall))

	print('Average precision: '+str(np.mean(average_precisions)))
	print('Average recall: '+str(np.mean(average_recalls)))

	return np.mean(average_precisions), np.mean(average_recalls), precisions_at_k, recalls_at_k, average_precisions, average_recalls, sorted_scores[:900]

if __name__ == '__main__':
	populate_image_to_index()

	image_files = os.listdir(IMAGES_DIR)
	# labels, centroids = quantize_colour_space()

	# centroids = np.uint8(centroids)
	# labels = labels.reshape((64, 64, 64, 1))

	# all_image_features = []

	# for i in range(len(image_files)):
	# for i in range(4000, len(image_files)):	
	# for i in range(3):
		# hist = autoCorrelogram(read_image(i), centroids, labels)
		# all_image_features.append(hist)
		# print(i)

	# pickle.dump(all_image_features, open('color_autocorrelogram_features_5.pickle', 'wb'))
	# final_features = merge_pickle_files()
	# pickle.dump(final_features, open('final_autocorrelogram_features.pickle', 'wb'))

	all_image_features = pickle.load(open('final_autocorrelogram_features.pickle', 'rb'))
	queries, query_files = return_query_list()

	average_precisions = []
	average_recalls = []

	percentages_good = []
	percentages_ok = []
	percentages_junk = []

	precisions_at_k = {}
	recalls_at_k = {}

	for threshold in [100, 200, 300, 400, 500, 600, 700, 800, 900]:
		precisions_at_k[threshold] = []
		recalls_at_k[threshold] = []

	start_time = time.time()

	for i, q in enumerate(queries):
		print('---------------------------------------')
		print('Processing query image: '+str(q)+', completed: '+str(i)+'/'+str(len(queries)))
		query_file = query_files[i]
		query_img_index = IMAGE_TO_INDEX_DICT[q]
		query_feature = all_image_features[IMAGE_TO_INDEX_DICT[q]]
		ground_truths = return_ground_truths(query_file)
		avg_precision, avg_recall, p_at_k, r_at_k, _, _, matching_results = matching(query_feature, query_img_index, all_image_features, [100, 200, 300, 400, 500, 600, 700, 800, 900], ground_truths)
		print('---------------------------------------')
		
		percent_good, percent_ok, percent_junk = return_retrieval_percentages(matching_results, query_file)

		percentages_good.append(percent_good)
		percentages_ok.append(percent_ok)
		percentages_junk.append(percent_junk)

		average_precisions.append(avg_precision)
		average_recalls.append(avg_recall)

		for threshold in [100, 200, 300, 400, 500, 600, 700, 800, 900]:
			precisions_at_k[threshold].append(p_at_k[threshold])
			recalls_at_k[threshold].append(r_at_k[threshold])

	print('--------------------------------------------------------')

	avg_precision = np.mean(average_precisions)
	recall = np.mean(average_recalls)

	end_time = time.time()

	print('Average precision: '+str(avg_precision))
	print('Recall: '+str(recall))
	print('F1 score: '+str(2*avg_precision*recall/(avg_precision+recall)))

	print('Max precision: '+str(np.max(average_precisions)))
	print('Min precision: '+str(np.min(average_precisions)))

	for threshold in [100, 200, 300, 400, 500, 600, 700, 800, 900]:
		print('Average precision @ '+str(threshold)+': '+str(np.mean(precisions_at_k[threshold])))						
		print('Average recall @ '+str(threshold)+': '+str(np.mean(recalls_at_k[threshold])))						

	print('Average time per retrieval: '+str((end_time-start_time)/len(queries)))

	print('Average percentage of good: '+str(np.mean(percentages_good)))
	print('Average percentage of ok: '+str(np.mean(percentages_ok)))
	print('Average percentage of junk: '+str(np.mean(percentages_junk)))