import nltk
from nltk.corpus import abc
import numpy as np
from nltk.tokenize import word_tokenize
import torch
from torch.autograd import Variable
import torch.nn as nn
import pickle
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy import spatial

from q1 import CBOW_model, SkipGram_model

words = ['landscape', 'music', 'dinosaur', 'male', 'throw', 'human', 'baby', 'dog', 'design', 'hardware', 'support', 'cancer', 'environment', 'flight', 'AIDS', 'report']

############################# Evaluate CBOW model without subsampling data ########################################

train_set = pickle.load(open('cbow_without_subsampling_data.pickle', 'rb'))

vocab, w2i, i2w, training_data = train_set['vocab'], train_set['w2i'], train_set['i2w'], train_set['training_data']

print(len(vocab), len(i2w), len(training_data))

model = CBOW_model(len(vocab), 50)
model.load_state_dict(torch.load('cbow_without_subsampling.ckpt'))

with torch.no_grad():
	all_embeddings = []

	for i in range(len(vocab)):
		emb = model.get_embeddings(Variable(torch.LongTensor([i])))
		all_embeddings.append(emb.cpu().numpy())

	all_embeddings = np.asarray(all_embeddings).reshape((len(vocab), 50))
	print(all_embeddings.shape)

	for word in words:
		print('Current word: '+word)

		word_ind = w2i[word]
		sim = []	
		vec = all_embeddings[word_ind]
		
		for ind, vecs in enumerate(all_embeddings):
			curr_vec = vecs
			result = 1 - spatial.distance.cosine(vec, curr_vec)
			sim.append((result, i2w[ind]))

		print(sorted(sim, reverse=True)[:10])

############################# Evaluate CBOW model with subsampling data ########################################

train_set = pickle.load(open('cbow_with_subsampling_data.pickle', 'rb'))

vocab, w2i, i2w, training_data = train_set['vocab'], train_set['w2i'], train_set['i2w'], train_set['training_data']

print(len(vocab), len(i2w), len(training_data))

model = CBOW_model(len(vocab), 50)
model.load_state_dict(torch.load('cbow_with_subsampling.ckpt'))

with torch.no_grad():
	all_embeddings = []

	for i in range(len(vocab)):
		emb = model.get_embeddings(Variable(torch.LongTensor([i])))
		all_embeddings.append(emb.cpu().numpy())

	all_embeddings = np.asarray(all_embeddings).reshape((len(vocab), 50))
	print(all_embeddings.shape)

	for word in words:
		print('Current word: '+word)

		word_ind = w2i[word]
		sim = []	
		vec = all_embeddings[word_ind]
		
		for ind, vecs in enumerate(all_embeddings):
			curr_vec = vecs
			result = 1 - spatial.distance.cosine(vec, curr_vec)
			sim.append((result, i2w[ind]))

		print(sorted(sim, reverse=True)[:10])

############################# Evaluate SkipGram model with subsampling data ########################################

train_set = pickle.load(open('skipgram_with_subsampling_data.pickle', 'rb'))

vocab, w2i, i2w, training_data = train_set['vocab'], train_set['w2i'], train_set['i2w'], train_set['training_data']

print(len(vocab), len(i2w), len(training_data))

model = SkipGram_model(len(vocab), 50)
model.load_state_dict(torch.load('skipgram_with_subsampling.ckpt'))

with torch.no_grad():
	all_embeddings = []

	for i in range(len(vocab)):
		emb = model.get_embeddings(Variable(torch.LongTensor([i])))
		all_embeddings.append(emb.cpu().numpy())

	all_embeddings = np.asarray(all_embeddings).reshape((len(vocab), 50))
	print(all_embeddings.shape)

	for word in words:
		print('Current word: '+word)

		word_ind = w2i[word]
		sim = []	
		vec = all_embeddings[word_ind]
		
		for ind, vecs in enumerate(all_embeddings):
			curr_vec = vecs
			result = 1 - spatial.distance.cosine(vec, curr_vec)
			sim.append((result, i2w[ind]))

		print(sorted(sim, reverse=True)[:10])

# ############################# Evaluate SkipGram model without subsampling data ########################################

train_set = pickle.load(open('skipgram_without_subsampling_data.pickle', 'rb'))

vocab, w2i, i2w, training_data = train_set['vocab'], train_set['w2i'], train_set['i2w'], train_set['training_data']

print(len(vocab), len(i2w), len(training_data))

model = SkipGram_model(len(vocab), 50)
model.load_state_dict(torch.load('skipgram_without_subsampling.ckpt'))

with torch.no_grad():
	all_embeddings = []

	for i in range(len(vocab)):
		emb = model.get_embeddings(Variable(torch.LongTensor([i])))
		all_embeddings.append(emb.cpu().numpy())

	all_embeddings = np.asarray(all_embeddings).reshape((len(vocab), 50))
	print(all_embeddings.shape)

	for word in words:
		print('Current word: '+word)

		word_ind = w2i[word]
		sim = []	
		vec = all_embeddings[word_ind]
		
		for ind, vecs in enumerate(all_embeddings):
			curr_vec = vecs
			result = 1 - spatial.distance.cosine(vec, curr_vec)
			sim.append((result, i2w[ind]))

		print(sorted(sim, reverse=True)[:10])