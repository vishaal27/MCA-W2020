import os
import numpy as np
import matplotlib.pyplot as plt
import wave
import pylab

labels_encoder = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}

def read_file(wav_file):
	wav = wave.open(wav_file, 'r')
	frames = wav.readframes(-1)
	file_series = pylab.fromstring(frames, 'Int16')
	frame_rate = wav.getframerate()
	wav.close()
	return file_series, frame_rate

def get_fourier_coeff_at_n(signal, curr_time_sample):
	signal_length = len(signal)
	all_freqs = np.arange(0, signal_length, 1)
	coeff = np.sum(signal * np.exp((1j*2*np.pi*all_freqs*curr_time_sample)/signal_length))/signal_length
	return coeff

def get_fourier_coefficients(signal):
	coefficients = []
	signal_length = len(signal)
	
	for n in range(signal_length//2): # Nyquest Limit
		coefficients.append(np.abs(get_fourier_coeff_at_n(signal, n))*2)
	return np.asarray(coefficients)

def inbuilt_spectrogram(signal, sampling_rate, window_size, noverlap, class_label):
	plt.specgram(signal, NFFT=window_size, Fs=sampling_rate, noverlap=noverlap)
	plt.savefig('./plots_spectrogram/inbuilt_'+class_label+'.png')

def spectrogram(signal, sampling_rate, window_size, num_overlap):
	signal_length = len(signal)
	time_samples = np.arange(0, signal_length, window_size-num_overlap, dtype=int)

	fourier_coeffs = []
	
	for n in time_samples:
		window_coeffs = get_fourier_coefficients(signal[n:n+window_size]) 
		fourier_coeffs.append(window_coeffs)

	log_spectrogram = 10*np.log10(np.transpose(np.asarray(fourier_coeffs[:-1])).astype(float))
	return log_spectrogram

def plot_spectrogram(spectrogram, class_label):
	print(spectrogram.shape)
	plt.imshow(spectrogram, origin='lower')

	plt.ylabel("Frequency")
	plt.xlabel("Time")
	plt.title("Spectrogram")
	plt.savefig('./plots_spectrogram/implemented_'+class_label+'.png')

if __name__ == '__main__':
	sampling_rate = 16000

	if not os.path.exists('./plots_spectrogram'):
		os.makedirs('./plots_spectrogram')

	wav_files = ['./training/zero/0b09edd3_nohash_0.wav', './training/one/1cec8d71_nohash_0.wav', './training/two/00b01445_nohash_2.wav', './training/three/0a9f9af7_nohash_0.wav', './training/four/0a9f9af7_nohash_0.wav', './training/five/0a9f9af7_nohash_0.wav', './training/six/0a7c2a8d_nohash_0.wav', './training/seven/0a0b46ae_nohash_0.wav', './training/eight/004ae714_nohash_0.wav', './training/nine/0a7c2a8d_nohash_0.wav']

	for idx, wav_file in enumerate(wav_files):
		print('Processing '+wav_file)
		file_series, frame_rate = read_file(wav_file)
		file_series = list(np.pad(np.asarray(file_series), (0, sampling_rate-len(file_series)), mode='constant', constant_values=0))
		
		inbuilt_spectrogram(file_series, sampling_rate, 128, 0, labels_encoder[idx])
		log_spectrogram = spectrogram(file_series, sampling_rate, 128, 0)
		plot_spectrogram(log_spectrogram, labels_encoder[idx])
