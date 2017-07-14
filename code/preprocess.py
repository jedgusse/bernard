#!/usr/bin/env

from collections import namedtuple as nt
from cltk.tokenize.sentence import TokenizeSentence
import itertools
import glob
import numpy as np
import random
import re
from scipy import stats
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import (Normalizer,
                                   StandardScaler,
                                   FunctionTransformer)
import sys
import time
import threading

def deltavectorizer(X):
	    # "An expression of pure difference is what we need"
	    #  Burrows' Delta -> Absolute Z-scores
	    X = np.abs(stats.zscore(X))
	    X = np.nan_to_num(X)
	    return X

def randomizer(authors, titles, texts, sample_size, 
			   test_dict, n_samples, smooth_test):

	""" |--- Function for randomly sampling from texts ---|
		::: Authors, Titles, Texts ::: """

	sampled_authors = []
	sampled_titles = []
	sampled_texts = []

	# Make train-test dict
	# Texts under the same author name are collected in one pool and then randomized
	pooled_dict = {author: [] for author in authors}
	for author, title, text in zip(authors, titles, texts):
		if author in pooled_dict:
			pooled_dict[author].append((title, text))

	# Instantiate cltk Tokenizer
	tokenizer = TokenizeSentence('latin')

	for author in pooled_dict:
		# Pool together texts by same author
		pooled_titles = [tup[0] for tup in pooled_dict[author]]
		pooled_texts = [tup[1] for tup in pooled_dict[author]]

		if author in test_dict and test_dict[author] in pooled_titles and smooth_test == False:
			print("::: test set «{} {}» is sampled in ordinary slices :::".format(author, "+".join(pooled_titles)))
			bulk = []
			for ord_text in pooled_texts:
				for word in ord_text.strip().split():
					word = word.lower()
					word = "".join([char for char in word if char not in punctuation])
					word = word.lower()
					bulk.append(word)
				# Safety measure against empty strings in samples
				bulk = [word for word in bulk if word != ""]
				bulk = [bulk[i:i+sample_size] for i in range(0, len(bulk), sample_size)]
				for index, sample in enumerate(bulk):
					if len(sample) == sample_size: 
						sampled_authors.append(author)
						sampled_titles.append(test_dict[author] + "_{}".format(str(index + 1)))
						sampled_texts.append(" ".join(sample))

		else:
			# Make short random samples and add to sampled texts
			# Remove punctuation in the meantime
			print("::: training set «{} {}» is randomly sampled from corpus :::".format(author, "+".join(pooled_titles)))
			pooled_texts = " ".join(pooled_texts)
			pooled_texts = tokenizer.tokenize_sentences(pooled_texts)
			if len(pooled_texts) < 20:
				print("-----| ERROR: please check if input texts have punctuation, tokenization returned only {} sentence(s) |-----".format(len(pooled_texts)))
				break
			for _ in range(1, n_samples+1):
				random_sample = []
				while len(" ".join(random_sample).split()) <= sample_size:
					random_sample.append(random.choice(pooled_texts))
				for index, word in enumerate(random_sample):
					random_sample[index] = "".join([char for char in word if char not in punctuation])
				random_sample = " ".join(random_sample).split()[:sample_size]
				sampled_authors.append(author)
				sampled_titles.append('sample_{}'.format(_))
				sampled_texts.append(" ".join(random_sample))

	return sampled_authors, sampled_titles, sampled_texts

class DataReader:

	""" |--- Defines metadata ---|
		::: Authors, Titles, Texts ::: """

	def __init__(self, folder_location, sample_size, test_dict, rnd_dict):
		self.folder_location = folder_location
		self.sample_size = sample_size
		self.test_dict = test_dict
		self.rnd_dict = rnd_dict

	def metadata(self, sampling, type, randomization):

		authors = []
		titles = []
		texts = []

		""" |--- Accepts both entire folders as files 
		::: More flexibility ---|"""

		if type == 'folder':
		
			for filename in glob.glob(self.folder_location + "/*"):
				author = filename.split("/")[-1].split(".")[0].split("_")[0]
				title = filename.split("/")[-1].split(".")[0].split("_")[1]

				bulk = []

				fob = open(filename)
				text = fob.read()
				for word in text.strip().split():
					if randomization == True:
						word = word.lower()
					else:
						word = "".join([char for char in word if char not in punctuation])
						word = word.lower()
					bulk.append(word)
				# Safety measure against empty strings in samples
				bulk = [word for word in bulk if word != ""]

				if sampling == True:
					if randomization == True:
						print("-- | ERROR: randomization and sampling both set to True")
						break
					else:
						bulk = [bulk[i:i+self.sample_size] for i in range(0, len(bulk), self.sample_size)]
						for index, sample in enumerate(bulk):
							if len(sample) == self.sample_size:
								authors.append(author)
								titles.append(title + "_{}".format(str(index + 1)))
								texts.append(" ".join(sample))

				elif sampling == False:
					authors.append(author)
					titles.append(title)
					bulk = " ".join(bulk)
					texts.append(bulk)

			if randomization == True:
				authors, titles, texts = randomizer(authors, titles, texts,
									   	 self.sample_size, self.test_dict, 
									   	 n_samples=self.rnd_dict['n_samples'],
									   	 smooth_test=self.rnd_dict['smooth_test'])

			return authors, titles, texts

		elif type == 'file':

			# The input is not a folder location, but a filename
			# So change the variable name from here on out

			filename = self.folder_location

			bulk = []

			fob = open(filename)
			author = filename.split("/")[-1].split(".")[0].split("_")[0]
			title = filename.split("/")[-1].split(".")[0].split("_")[1]
			text = fob.read()
			for word in text.strip().split():
				word = [char for char in word if char not in punctuation]
				word = "".join(word)
				word = word.lower()
				bulk.append(word)
			# Safety measure against empty strings in samples
			bulk = [word for word in bulk if word != ""]

			if sampling == True:
				bulk = [bulk[i:i+self.sample_size] for i in range(0, len(bulk), self.sample_size)]

				for index, sample in enumerate(bulk):
					if len(sample) == self.sample_size:
						authors.append(author)
						titles.append(title + "_{}".format(str(index + 1)))
						texts.append(" ".join(sample))

			else:
				texts = " ".join(bulk)

			return author, title, texts

class Vectorizer:

	""" |--- From flat text to document vectors ---|
		::: Document Vectors, Most Common Features ::: """

	def __init__(self, texts, stop_words, n_feats, feat_scaling, analyzer, vocab):
		self.texts = texts
		self.stop_words = stop_words
		self.n_feats = n_feats
		self.feat_scaling = feat_scaling
		self.analyzer = analyzer
		self.vocab = vocab
		self.norm_dict = {'delta': FunctionTransformer(deltavectorizer), 
				   	   	  'normalizer': Normalizer(),
						  'standard_scaler': StandardScaler()}

	# Raw Vectorization

	def raw(self):

		# Text vectorization; array reversed to order of highest frequency
		# Vectorizer takes a list of strings

		# Define fed-in analyzer
		ngram_range = None
		if self.analyzer == 'char':
			ngram_range = ((2,4))
		elif self.analyzer == 'word':
			ngram_range = ((1,1))

		model = CountVectorizer(stop_words=self.stop_words, 
							    max_features=self.n_feats,
							    analyzer=self.analyzer,
							    vocabulary=self.vocab,
							    ngram_range=ngram_range)
		doc_vectors = model.fit_transform(self.texts).toarray()
		corpus_vector = np.ravel(np.sum(doc_vectors, axis=0))
		features = model.get_feature_names()
		mfwords = [x for (y,x) in sorted(zip(corpus_vector, features), reverse=True)]

		new_X = []
		for word in mfwords:
			for feature, freq in zip(features, doc_vectors.transpose()):
				if word == feature:
					new_X.append(freq)
		doc_vectors = np.array(new_X).transpose()

		if self.feat_scaling == False:
			pass
		else:
			doc_vectors = self.norm_dict[self.feat_scaling].fit_transform(doc_vectors)

		return doc_vectors, mfwords

	# Term-Frequency Inverse Document Frequency Vectorization

	def tfidf(self, smoothing):

		# Define fed-in analyzer
		ngram_range = None
		if self.analyzer == 'char':
			ngram_range = ((2,4))
		elif self.analyzer == 'word':
			ngram_range = ((1,1))

		# Text vectorization; array reversed to order of highest frequency
		model = TfidfVectorizer(stop_words=self.stop_words, 
							    max_features=self.n_feats,
							    smooth_idf=smoothing,
							    analyzer=self.analyzer,
							    vocabulary=self.vocab,
							    ngram_range=ngram_range)
		tfidf_vectors = model.fit_transform(self.texts).toarray()
		corpus_vector = np.ravel(np.sum(tfidf_vectors, axis=0))
		features = model.get_feature_names()
		mfwords = [x for (y,x) in sorted(zip(corpus_vector, features), reverse=True)]

		new_X = []
		for word in mfwords:
			for feature, freq in zip(features, tfidf_vectors.transpose()):
				if word == feature:
					new_X.append(freq)
		tfidf_vectors = np.array(new_X).transpose()

		if self.feat_scaling == False:
			pass
		else:
			tfidf_vectors = self.norm_dict[self.feat_scaling].fit_transform(tfidf_vectors)
			
		return tfidf_vectors, mfwords

