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

COLOUR_NAMES=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

class CBOW_model(nn.Module):
	def __init__(self, vocab, dims):
		super(CBOW_model, self).__init__()
		self.embeddings = nn.Embedding(vocab, dims)
		self.linear = nn.Linear(dims, vocab)

	def get_embeddings(self, input):
		embeds = self.embeddings(input)
		return embeds

	def forward(self, input):
		embeds = torch.mean(self.embeddings(input), dim=0)
		out = self.linear(embeds.view((1, -1)))
		return F.log_softmax(out, dim=1)

class SkipGram_model(nn.Module):
	def __init__(self, vocab, dims):
		super(SkipGram_model, self).__init__()
		self.embeddings = nn.Embedding(vocab, dims)
		self.linear = nn.Linear(dims, vocab)

	def get_embeddings(self, input):
		embeds = self.embeddings(input)
		return embeds

	def forward(self, inputs):
		embeds = self.embeddings(inputs)
		out = self.linear(embeds.view((1, -1)))
		return F.log_softmax(out, dim=1)

def get_corpus():
	science = abc.raw('science.txt')
	rural = abc.raw('rural.txt')
	concat = science+'\n'+rural
	return concat

def create_mappings(processed_text, subsampling):
	vocab = {}
	i2w = {}
	w2i = {}

	total = 0
	
	for itr, word in enumerate(processed_text):
		if word not in vocab:
			vocab[word] = 0
			_ind = len(w2i)
			w2i[word] = _ind
			i2w[_ind] = word
			
		vocab[word] += 1
		total += 1

	if subsampling:
		for i, word in enumerate(processed_text):
			p_i = np.sqrt(0.001 * total / vocab[word]) + np.sqrt(0.001 * total / vocab[word])**2
			if (np.random.sample()<=p_i):

				del [processed_text[i]]
				i -= 1

	return vocab, w2i, i2w, processed_text

def check_cond_1(window, end_pos_1, start_pos_1):
	return window>end_pos_1 and start_pos_1>=0

def check_cond_2(window, end_pos_2, processed_text, start_pos_2):
	return window>end_pos_2 and start_pos_2<len(processed_text)

def create_skipgram_data(processed_text, window, w2i):
	data = []

	for i, word in enumerate(processed_text):
		start_pos_1 = i-1
		start_pos_2 = start_pos_1+2

		end_pos_1 = 0
		end_pos_2 = end_pos_1
		
		while(check_cond_1(window, end_pos_1, start_pos_1)):
			target = w2i[processed_text[start_pos_1]]
			context = list([w2i[word]])
			data.append((context, target))
			start_pos_1-=1
			end_pos_1+=1
		
		while(check_cond_2(window, end_pos_2, processed_text, start_pos_2)):
			target = w2i[processed_text[start_pos_2]]
			context = list([w2i[word]])
			data.append((context, target))
			start_pos_2+=1
			end_pos_2+=1

	return data

def create_cbow_data(processed_text, window, w2i):
	data = []

	for i, word in enumerate(processed_text):
		start_pos_1 = i-1
		start_pos_2 = start_pos_1+2

		end_pos_1 = 0
		end_pos_2 = end_pos_1

		cbow_context = []

		while(check_cond_1(window, end_pos_1, start_pos_1)):
			curr_context = w2i[processed_text[start_pos_1]]
			cbow_context.append(curr_context)
			start_pos_1-=1
			end_pos_1+=1
		while(check_cond_2(window, end_pos_2, processed_text, start_pos_2)):
			curr_context = w2i[processed_text[start_pos_2]]
			cbow_context.append(curr_context)
			start_pos_2+=1
			end_pos_2+=1

		target = w2i[word]
		data.append((cbow_context, target))
	
	return data

def create_train_data(processed_text, window, w2i, model):
	training_data = []
	if(model=="skipgram"):
		training_data = create_skipgram_data(processed_text, window, w2i)
	else:
		training_data = create_cbow_data(processed_text, window, w2i)
	return training_data

def dataloader(corpus, window, model, subsampling):
	print('Tokenizing data...')
	processed_text = word_tokenize(corpus.strip())
	print('Building vocabulary...')
	vocab, w2i, i2w, processed_text = create_mappings(processed_text, subsampling)
	print('Creating training data...')
	training_data = create_train_data(processed_text, window, w2i, model)
	return vocab, w2i, i2w, training_data

if __name__ == '__main__':

	# Compute and store training sets
	# corpus = get_corpus()
	# vocab, w2i, i2w, training_data = dataloader(corpus, 2, 'skipgram', True)
	# pickle.dump({'vocab': vocab, 'w2i': w2i, 'i2w': i2w, 'training_data': training_data}, open('skipgram_with_subsampling_data.pickle', 'wb'))

	# vocab, w2i, i2w, training_data = dataloader(corpus, 2, 'cbow', True)
	# pickle.dump({'vocab': vocab, 'w2i': w2i, 'i2w': i2w, 'training_data': training_data}, open('cbow_with_subsampling_data.pickle', 'wb'))
	
	# vocab, w2i, i2w, training_data = dataloader(corpus, 2, 'skipgram', False)
	# pickle.dump({'vocab': vocab, 'w2i': w2i, 'i2w': i2w, 'training_data': training_data}, open('skipgram_without_subsampling_data.pickle', 'wb'))

	# vocab, w2i, i2w, training_data = dataloader(corpus, 2, 'cbow', False)
	# pickle.dump({'vocab': vocab, 'w2i': w2i, 'i2w': i2w, 'training_data': training_data}, open('cbow_without_subsampling_data.pickle', 'wb'))
	
	############################# Train CBOW model without subsampling data ########################################

	train_set = pickle.load(open('cbow_without_subsampling_data.pickle', 'rb'))

	vocab, w2i, i2w, training_data = train_set['vocab'], train_set['w2i'], train_set['i2w'], train_set['training_data']

	random.shuffle(training_data)
	train_data, test_data = training_data[:int(0.8*len(training_data))], training_data[int(0.8*len(training_data)):]
	print(len(train_data), len(test_data))

	loss_ = nn.NLLLoss()
	# model = SkipGram_model(len(vocab), 50)
	model = CBOW_model(len(vocab), 50)
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
	total_losses = []
 
	print("Starting training for CBOW without subsampling with "+str(len(train_data))+" samples")
	for epoch in range(20):
		for itr in range(50):
			total_loss = torch.zeros(1)
			train_slice = train_data
			random.shuffle(train_slice)

			for _,(context, target) in enumerate(train_slice[:1000]):
				context_var = Variable(torch.LongTensor(context))
				log_probs = model(context_var)
				loss = loss_(log_probs, Variable(torch.LongTensor([target])))
				model.zero_grad()
				loss.backward()
				optimizer.step()
				total_loss += loss.item()
			print("Epoch "+str(epoch)+", Iteration: "+str(itr)+", Loss: "+str(total_loss.item()))
			total_losses.append(total_loss)

		with torch.no_grad():
			test_embeddings = []
			words = ['landscape', 'music', 'dinosaur', 'male', 'throw', 'human', 'baby', 'dog', 'design', 'hardware', 'support', 'cancer', 'environment', 'flight', 'AIDS', 'report']
			
			for word_ind in range(4):
				context = [w2i[words[word_ind*4]], w2i[words[word_ind*4+1]], w2i[words[word_ind*4+2]], w2i[words[word_ind*4+3]]]
				embs = model.get_embeddings(Variable(torch.LongTensor(context)))
				for emb in embs:
					test_embeddings.append(emb.cpu().numpy())

			tsne = TSNE(n_components=2, perplexity=100).fit_transform(test_embeddings)

			plt.figure()
			for i in range(len(tsne)):
				plt.scatter(tsne[i][0], tsne[i][1], s=4, label=words[i])
			plt.legend(loc='best')
			plt.title('words TSNE for epoch_'+str(epoch))
			plt.savefig('./tsne_plots/cbow_without_subsampling/words_epoch_'+str(epoch)+'.png')

			for context, _ in test_data[:100]:
				embs = model.get_embeddings(Variable(torch.LongTensor(context)))
				for emb in embs:
					test_embeddings.append(emb.cpu().numpy())
			tsne = TSNE(n_components=2, perplexity=100).fit_transform(test_embeddings)

			plt.figure()
			for i in range(len(tsne)):
				plt.scatter(tsne[i][0], tsne[i][1], s=3)
			plt.title('TSNE for epoch_'+str(epoch))
			plt.savefig('./tsne_plots/cbow_without_subsampling/epoch_'+str(epoch)+'.png')
			plt.close()

	torch.save(model.state_dict(), 'cbow_without_subsampling.ckpt')

	############################ Train CBOW model with subsampling data ##########################################

	train_set = pickle.load(open('cbow_with_subsampling_data.pickle', 'rb'))

	vocab, w2i, i2w, training_data = train_set['vocab'], train_set['w2i'], train_set['i2w'], train_set['training_data']

	random.shuffle(training_data)
	train_data, test_data = training_data[:int(0.8*len(training_data))], training_data[int(0.8*len(training_data)):]
	print(len(train_data), len(test_data))

	loss_ = nn.NLLLoss()
	# model = SkipGram_model(len(vocab), 50)
	model = CBOW_model(len(vocab), 50)
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
	total_losses = []
 
	print("Starting training for CBOW with subsampling with "+str(len(train_data))+" samples")
	for epoch in range(20):
		for itr in range(50):
			total_loss = torch.zeros(1)
			train_slice = train_data
			random.shuffle(train_slice)

			for _,(context, target) in enumerate(train_slice[:1000]):
				context_var = Variable(torch.LongTensor(context))
				log_probs = model(context_var)
				loss = loss_(log_probs, Variable(torch.LongTensor([target])))
				model.zero_grad()
				loss.backward()
				optimizer.step()
				total_loss += loss.item()
			print("Epoch "+str(epoch)+", Iteration: "+str(itr)+", Loss: "+str(total_loss.item()))
			total_losses.append(total_loss)

		with torch.no_grad():
			test_embeddings = []
			words = ['landscape', 'music', 'dinosaur', 'male', 'throw']
			
			for word_ind in range(4):
				context = [w2i[words[word_ind*4]], w2i[words[word_ind*4+1]], w2i[words[word_ind*4+2]], w2i[words[word_ind*4+3]]]
				embs = model.get_embeddings(Variable(torch.LongTensor(context)))
				for emb in embs:
					test_embeddings.append(emb.cpu().numpy())

			tsne = TSNE(n_components=2, perplexity=100).fit_transform(test_embeddings)

			plt.figure()
			for i in range(len(tsne)):
				plt.scatter(tsne[i][0], tsne[i][1], s=4, label=words[i])
			plt.legend(loc='best')
			plt.title('words TSNE for epoch_'+str(epoch))
			plt.savefig('./tsne_plots/cbow_with_subsampling/words_epoch_'+str(epoch)+'.png')
			plt.close()

			for context, _ in test_data[:100]:
				embs = model.get_embeddings(Variable(torch.LongTensor(context)))
				for emb in embs:
					test_embeddings.append(emb.cpu().numpy())
			tsne = TSNE(n_components=2, perplexity=100).fit_transform(test_embeddings)

			plt.figure()
			for i in range(len(tsne)):
				plt.scatter(tsne[i][0], tsne[i][1], s=3)
			plt.title('TSNE for epoch_'+str(epoch))
			plt.savefig('./tsne_plots/cbow_with_subsampling/epoch_'+str(epoch)+'.png')
			plt.close()

			all_embeddings = []

			for i in range(len(vocab)):
				emb = model.get_embeddings(Variable(torch.LongTensor([i])))
				all_embeddings.append(emb.cpu().numpy())

			all_embeddings = np.asarray(all_embeddings).reshape((len(vocab), 50))

			land_arr = [all_embeddings[w2i['landscape']]]
			music_arr = [all_embeddings[w2i['music']]]
			dino_arr = [all_embeddings[w2i['dinosaur']]]
			male_arr = [all_embeddings[w2i['male']]]
			throw_arr = [all_embeddings[w2i['throw']]]

			all_arrs = [land_arr, music_arr, dino_arr, male_arr, throw_arr]

			for i, word in enumerate(words):
				word_ind = w2i[word]
				sim = []	
				vec = all_embeddings[word_ind]
				
				for ind, vecs in enumerate(all_embeddings):
					curr_vec = vecs
					result = 1 - spatial.distance.cosine(vec, curr_vec)
					sim.append((result, ind))

				top_similar = [x[1] for x in sorted(sim, reverse=True)[:5]]
				top_embeds = [all_embeddings[i] for i in top_similar]
				
				all_arrs[i] = all_arrs[i]+top_embeds

			concat = all_arrs[0]+all_arrs[1]+all_arrs[2]+all_arrs[3]+all_arrs[4]
			tsne = TSNE(n_components=2, perplexity=100).fit_transform(concat)
			print(tsne.shape)
			plt.figure()
			for i in range(len(tsne)):
				plt.scatter(tsne[i][0], tsne[i][1], s=4, c=COLOUR_NAMES[i//6])
			plt.title('TSNE for epoch_'+str(epoch))
			plt.savefig('./tsne_plots/cbow_with_subsampling/similar_epoch_'+str(epoch)+'.png')
			plt.close()

	torch.save(model.state_dict(), 'cbow_with_subsampling.ckpt')

	############################ Train Skipgram model without subsampling data ##########################################
	
	train_set = pickle.load(open('skipgram_without_subsampling_data.pickle', 'rb'))

	vocab, w2i, i2w, training_data = train_set['vocab'], train_set['w2i'], train_set['i2w'], train_set['training_data']

	random.shuffle(training_data)
	train_data, test_data = training_data[:int(0.8*len(training_data))], training_data[int(0.8*len(training_data)):]
	print(len(train_data), len(test_data))

	loss_ = nn.NLLLoss()
	model = SkipGram_model(len(vocab), 50)
	# model = CBOW_model(len(vocab), 50)
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
	total_losses = []
 
	print("Starting training for Skipgram without subsampling with "+str(len(train_data))+" samples")
	for epoch in range(20):
		for itr in range(50):
			total_loss = torch.zeros(1)
			train_slice = train_data
			random.shuffle(train_slice)

			for _,(context, target) in enumerate(train_slice[:1000]):
				context_var = Variable(torch.LongTensor(context))
				log_probs = model(context_var)
				loss = loss_(log_probs, Variable(torch.LongTensor([target])))
				model.zero_grad()
				loss.backward()
				optimizer.step()
				total_loss += loss.item()
			print("Epoch "+str(epoch)+", Iteration: "+str(itr)+", Loss: "+str(total_loss.item()))
			total_losses.append(total_loss)

		with torch.no_grad():
			test_embeddings = []
			words = ['landscape', 'music', 'dinosaur', 'male', 'throw', 'human', 'baby', 'dog', 'design', 'hardware', 'support', 'cancer', 'environment', 'flight', 'AIDS', 'report']
			
			for word_ind in range(4):
				context = [w2i[words[word_ind*4]], w2i[words[word_ind*4+1]], w2i[words[word_ind*4+2]], w2i[words[word_ind*4+3]]]
				embs = model.get_embeddings(Variable(torch.LongTensor(context)))
				for emb in embs:
					test_embeddings.append(emb.cpu().numpy())

			tsne = TSNE(n_components=2, perplexity=100).fit_transform(test_embeddings)

			plt.figure()
			for i in range(len(tsne)):
				plt.scatter(tsne[i][0], tsne[i][1], s=4, label=words[i])
			plt.legend(loc='best')
			plt.title('words TSNE for epoch_'+str(epoch))
			plt.savefig('./tsne_plots/skipgram_without_subsampling/words_epoch_'+str(epoch)+'.png')

			for context, _ in test_data[:100]:
				embs = model.get_embeddings(Variable(torch.LongTensor(context)))
				for emb in embs:
					test_embeddings.append(emb.cpu().numpy())
			tsne = TSNE(n_components=2, perplexity=100).fit_transform(test_embeddings)

			plt.figure()
			for i in range(len(tsne)):
				plt.scatter(tsne[i][0], tsne[i][1], s=3)
			plt.title('TSNE for epoch_'+str(epoch))
			plt.savefig('./tsne_plots/skipgram_without_subsampling/epoch_'+str(epoch)+'.png')
			plt.close()
	
	torch.save(model.state_dict(), 'skipgram_without_subsampling.ckpt')

	############################ Train Skipgram model with subsampling data ##########################################
	
	train_set = pickle.load(open('skipgram_with_subsampling_data.pickle', 'rb'))

	vocab, w2i, i2w, training_data = train_set['vocab'], train_set['w2i'], train_set['i2w'], train_set['training_data']

	random.shuffle(training_data)
	train_data, test_data = training_data[:int(0.8*len(training_data))], training_data[int(0.8*len(training_data)):]
	print(len(train_data), len(test_data))

	loss_ = nn.NLLLoss()
	model = SkipGram_model(len(vocab), 50)
	# model = CBOW_model(len(vocab), 50)
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
	total_losses = []
 
	print("Starting training for Skipgram with subsampling with "+str(len(train_data))+" samples")
	for epoch in range(20):
		for itr in range(50):
			total_loss = torch.zeros(1)
			train_slice = train_data
			random.shuffle(train_slice)

			for _,(context, target) in enumerate(train_slice[:1000]):
				context_var = Variable(torch.LongTensor(context))
				log_probs = model(context_var)
				loss = loss_(log_probs, Variable(torch.LongTensor([target])))
				model.zero_grad()
				loss.backward()
				optimizer.step()
				total_loss += loss.item()
			print("Epoch "+str(epoch)+", Iteration: "+str(itr)+", Loss: "+str(total_loss.item()))
			total_losses.append(total_loss)

		with torch.no_grad():
			test_embeddings = []
			words = ['landscape', 'music', 'dinosaur', 'male', 'throw', 'human', 'baby', 'dog', 'design', 'hardware', 'support', 'cancer', 'environment', 'flight', 'AIDS', 'report']
			
			for word_ind in range(4):
				context = [w2i[words[word_ind*4]], w2i[words[word_ind*4+1]], w2i[words[word_ind*4+2]], w2i[words[word_ind*4+3]]]
				embs = model.get_embeddings(Variable(torch.LongTensor(context)))
				for emb in embs:
					test_embeddings.append(emb.cpu().numpy())

			tsne = TSNE(n_components=2, perplexity=100).fit_transform(test_embeddings)

			plt.figure()
			for i in range(len(tsne)):
				plt.scatter(tsne[i][0], tsne[i][1], s=4, label=words[i])
			plt.legend(loc='best')
			plt.title('words TSNE for epoch_'+str(epoch))
			plt.savefig('./tsne_plots/skipgram_with_subsampling/words_epoch_'+str(epoch)+'.png')

			for context, _ in test_data[:100]:
				embs = model.get_embeddings(Variable(torch.LongTensor(context)))
				for emb in embs:
					test_embeddings.append(emb.cpu().numpy())
			tsne = TSNE(n_components=2, perplexity=100).fit_transform(test_embeddings)

			plt.figure()
			for i in range(len(tsne)):
				plt.scatter(tsne[i][0], tsne[i][1], s=3)
			plt.title('TSNE for epoch_'+str(epoch))
			plt.savefig('./tsne_plots/skipgram_with_subsampling/epoch_'+str(epoch)+'.png')
			plt.close()
	
	torch.save(model.state_dict(), 'skipgram_with_subsampling.ckpt')