#!/usr/bin/env

import os
from preprocess import DataReader, Vectorizer
from visualization import PrinCompAnal, GephiNetworks

# Insert folder where text files are located to perform:
	# Principal components analysis
	# k-NN networks (Gephi network analysis)
folder_location = os.path.dirname(os.getcwd()) + "/data"

# PARAMETERS
### ||| ------ ||| ###

sample_size = 3000
n_feats = 150

rnd_dct = {'n_samples': 140,
		   'smooth_train': True, 
		   'smooth_test': False}

invalid_words = ['dummyword']

# For classification tests, split data into training and test corpus (classifier will train and evaluate on training corpus, 
# and predict on new test corpus)

test_dict = {}

if __name__ == '__main__':

	print('::: preprocessing :::')
	authors, titles, texts = DataReader(folder_location, sample_size,
										test_dict, rnd_dct
										).metadata(sampling=True,
										type='folder',
										randomization=False)

	print('::: tfidf-vectorizing :::')
	tfidf_vectors, tfidf_features = Vectorizer(texts, invalid_words,
								  n_feats=n_feats,
								  feat_scaling='standard_scaler',
								  analyzer='word',
								  vocab=None
								  ).tfidf(smoothing=True)

	print('::: plotting principal components analysis :::')
	PrinCompAnal(authors, titles, tfidf_vectors, tfidf_features, sample_size, n_components=2).plot(
													show_samples=True,
													show_loadings=True,
													sbrn_plt=False)
	
	# Note that this returns node and edge worksheets (gephi_nodes.txt, gephi_edges.txt)
	# Gephi needs to be downloaded and the worksheets need to be imported
	# https://gephi.org/

	print('::: writing Gephi document sheets (nodes and edges) :::')
	GephiNetworks(folder_location, sample_size, invalid_words).plot(feat_range=[10, 50, 100, 150], 
																		random_sampling=None,
																		corpus_size=90)
	
