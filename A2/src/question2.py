import os
import numpy as np
import matplotlib.pyplot as plt
import wave
import pylab
from scipy.fftpack import dct
from python_speech_features import mfcc
import math
import sys

labels_encoder = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}

def read_file(wav_file):
	wav = wave.open(wav_file, 'r')
	frames = wav.readframes(-1)
	file_series = pylab.fromstring(frames, 'Int16')
	frame_rate = wav.getframerate()
	wav.close()
	return file_series, frame_rate

def inbuilt_mfcc(signal, sampling_rate, window_size, stride, nfilt, num_ceps, ceplifter, NFFT, class_label):
	mfcc_feats = mfcc(signal, sampling_rate, winlen=window_size, winstep=stride, numcep=num_ceps, nfilt=nfilt, nfft=NFFT, ceplifter=ceplifter)
	print(mfcc_feats.shape)
	plt.imshow(mfcc_feats.T, origin='lower')
	plt.savefig('./plots_mfcc/inbuilt_'+class_label+'.png')

def get_filter_banks(frames, nfilt, NFFT, bins):
	filter_bank = np.zeros((nfilt, int(np.floor((NFFT/2)+1))))
	
	for num_filt in range(1, nfilt+1):
		f_left, f_curr, f_right = int(bins[num_filt-1]), int(bins[num_filt]), int(bins[num_filt+1])
		for k in range(f_left, f_curr):
			num = k-bins[num_filt-1]
			denom = bins[num_filt]-bins[num_filt-1]
			filter_bank[num_filt-1, k] = num/denom

		for k in range(f_curr, f_right):
			num = bins[num_filt+1]-k
			denom = bins[num_filt+1]-bins[num_filt]
			filter_bank[num_filt-1, k] = num/denom

	filter_banks = 10*np.log10(np.dot(frames, filter_bank.T) + sys.float_info.epsilon)

	return filter_banks

def liftering_and_norm(mfcc_features, ceplifter):
	n = np.arange(0, mfcc_features.shape[1])
	lift = 1+(ceplifter/2)*np.sin(np.pi*n/ceplifter)
	mfcc_features *= lift
	mfcc_features -= (np.mean(mfcc_features))
	return mfcc_features

def compute_mfcc_feats(signal, sampling_rate, window_size, stride, nfilt, num_ceps, ceplifter, NFFT):
	window_size = int(window_size*sampling_rate)
	stride = int(stride*sampling_rate)
	signal_length = len(signal)

	n_dims = (signal_length - window_size)/stride
	if(n_dims%2==0):
		pass
	else:
		n_dims += 1

	num_frames = int(n_dims)
	signal = np.append(signal, np.zeros((window_size + num_frames*stride - signal_length))) 

	cur_index = np.tile(np.arange(0, window_size), (num_frames, 1)) + np.tile(np.arange(0, num_frames*stride, stride), (window_size, 1)).T
	frames = signal[cur_index]

	n = np.arange(0, window_size)
	frames *= 0.54-0.46*np.cos((2*np.pi*n) / (window_size-1))

	frames = np.abs(np.fft.rfft(frames, NFFT))
	frames = (1/NFFT)*(np.power(frames, 2))

	h_mel = (2595*np.log10(1+(sampling_rate)/1400))
	hz_points = (700*(10**(np.linspace(0, h_mel, nfilt+2)/2595)-1))

	mfcc_features = liftering_and_norm(dct(get_filter_banks(frames, nfilt, NFFT, np.floor((NFFT+1)*hz_points/sampling_rate)), axis=1)[:, 1:(num_ceps+1)], ceplifter)
	return mfcc_features

def plot_mfcc_features(mfcc_features, class_label):
	print(mfcc_features.shape)
	plt.imshow(mfcc_features.T, origin='lower')

	plt.ylabel("MFCC Coefficients")
	plt.xlabel("Time")
	plt.title("MFCC features")
	plt.savefig('./plots_mfcc/implemented_'+class_label+'.png')

if __name__ == '__main__':
	sampling_rate = 16000
	
	if not os.path.exists('./plots_mfcc'):
		os.makedirs('./plots_mfcc')

	wav_files = ['./training/zero/0b09edd3_nohash_0.wav', './training/one/1cec8d71_nohash_0.wav', './training/two/00b01445_nohash_2.wav', './training/three/0a9f9af7_nohash_0.wav', './training/four/0a9f9af7_nohash_0.wav', './training/five/0a9f9af7_nohash_0.wav', './training/six/0a7c2a8d_nohash_0.wav', './training/seven/0a0b46ae_nohash_0.wav', './training/eight/004ae714_nohash_0.wav', './training/nine/0a7c2a8d_nohash_0.wav']

	for idx, wav_file in enumerate(wav_files):
		print('Processing '+wav_file)
		file_series, frame_rate = read_file(wav_file)
		file_series = list(np.pad(np.asarray(file_series), (0, sampling_rate-len(file_series)), mode='constant', constant_values=0))
		signal = np.asarray(file_series, dtype=np.uint8)
		
		mfcc_features = compute_mfcc_feats(signal, sampling_rate, 0.025, 0.01, 40, 12, 20, 512)
		plot_mfcc_features(mfcc_features, labels_encoder[idx])
		inbuilt_mfcc(signal, sampling_rate, 0.025, 0.01, 40, 12, 20, 512, labels_encoder[idx])
