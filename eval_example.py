"""
Evaluation

Usage:
	eval_example.py [options] <checkpoint>

Options:
	--audio-sample=<path>			Path to an audio file.
	--disable-cuda						Disable CUDA.	
	-h, --help								Show help message.
"""
import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


TIME = 50
EMBEDDING_SIZE = 128
HIDDEN_LSTM_SIZE = 128
LSTM_LAYERS = 3


def get_args():
	"""Define and parse command-line arguments."""
	parser = argparse.ArgumentParser(description='Attention increasing or decreasing prediction' + \
		'based on audio features extracted on the wild from the Jameson short-film festival clips.')
	parser.add_argument(
		"checkpoint",
		type=str,
		help='Path from which restore checkpoint')
	parser.add_argument(
		"--audio-sample",
		type=str,
		help='Path to an audio file')
	parser.add_argument(
		"--disable-cuda",
		action='store_true',
		help='Disable CUDA')
	return parser.parse_args()


def generate_sample(time, embedding_size):
	"""Example of audio embedding"""
	embedding = np.random.randint(low=0,
		high=256,
		size=(1, time, embedding_size))
	return embedding


class LSTM(nn.Module):
	# TODO: Description
	def __init__(self, 
		input_size, 
		hidden_size, 
		output_size, 
		num_layers,
		device):
		super(LSTM, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.device = device

		self.lstm = nn.LSTM(input_size, hidden_size,
			num_layers=num_layers,
		 	batch_first=True)

		self.fc = nn.Linear(hidden_size, output_size)

	def forward(self, x):
		x_hat, (hidden, state) = self.lstm(x)
		x_hat = self.fc(x_hat[-1, -1]).reshape(-1)
		return torch.sigmoid(x_hat)


def load_checkpoint(model, save_dir):
	"""Load the model and training parameters.
		Args:
			model (nn.Module): Model to import.
			save_dir (str): Path to the model. 

		Returns:
		Model stored in the checkpoint.
	"""
	assert os.path.isfile(save_dir)
	assert save_dir.split('.')[1] == 'pth'
	checkpoint = torch.load(save_dir)
	model.load_state_dict(checkpoint['state_dict'])
	return model


if __name__ == "__main__":
	args = get_args()
	checkpoint_dir = args.checkpoint
	use_cuda = torch.cuda.is_available()

	# Manage CUDA devices:
	if use_cuda and not args.disable_cuda:
		device = torch.device('cuda:0')
	else:
		device = torch.device('cpu')
		use_cuda = False

	# Data
	if args.audio_sample is not None:
		# TODO: Change to audio sample and logmel generation
		embedding = np.expand_dims(np.load(args.audio_sample), axis=0)
		_, TIME, EMBEDDING_SIZE = embedding.shape
	else:
		embedding = generate_sample(TIME, EMBEDDING_SIZE)

	# Model
	model = LSTM(input_size=EMBEDDING_SIZE,
		hidden_size=HIDDEN_LSTM_SIZE,
		output_size=1, 
		num_layers=LSTM_LAYERS,
		device=device).to(device)
	model = load_checkpoint(model, checkpoint_dir)
	model.eval()

	# Predict attention on the available sample:
	with torch.no_grad():
		x = torch.from_numpy(embedding).float().to(device)
		x_hat = model(x)
		attention_level = 'increasing' if x_hat >= 0.5 else 'decreasing'

	print('Output of the network: {:.2f}\nThis audio clip is tagged with {} attention'.format(
		x_hat.data.cpu().numpy()[0], attention_level))


	
	