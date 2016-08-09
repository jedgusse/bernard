#!/usr/bin/env python

import glob
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import gensim, logging
import scipy.sparse
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance
import itertools
import matplotlib.patches as mpatches
from matplotlib import rc
import matplotlib.axes
import matplotlib.lines
from scipy.spatial.distance import cityblock as manhattan
from sklearn.neighbors import NearestNeighbors
import operator

# Switches that can be turned; PoS n-grams, permutations, lemma n-grams, length of text samples, ... Feel free to experiment.

n_pos = 3
n_perm = 3
n = 4
sample_length = 1500

# valid_postags decides which types of words are allowed to become features.
# In this case we have chosen only conjunctions, appositions, pronouns and adverbs to be taken into consideration. 

valid_postags = ["CON", "AP", "PRO", "ADV"]

# Invalid_words are those words which we have manually decided to leave out of our features list. 
# Some of these are a result of a tagging or lemmatization error, others are explained in the article.

invalid_words = ['alius', 'alter', 'certe', 'ego', 'eous', 'idem', 'libenter', 'meus', 'nos', 'noster', 'nullus', 'numquis', 
		         'sui', 'tu', 'tuus', 'vester', 'virum', 'vos', 'phoca', 'annuo', 'as', 'ianua', 'intervallum', 'lavo', 'v',
		         'accipio', 'ago', 'aio', 'audio', 'credo', 'debeo', 'dico', 'diligo', 'do', 'facio', 'fio', 'habeo', 'invenio',
		         'inquam', 'loquor', 'nescio', 'unquam', 'semper', 'amen', 'cito', 'numquam', 'interim', 'profectus', 'parve',
		         'pariter', 'annullo', 'averto', 'avis', 'cupio', 'festus', 'iucundus', 'milesius', 'moderor', 'nequeo', 'patrie',
		         'plivium', 'querela', 'tutus', 'utor', 'valeo', 'verecundus', 'aura', 'fecundus', 'talio', 'fideliter', 'huiusmodi',
		         'facile']
		         
# This is how the punctuation is tagged in the corpus. 
# We make sure no punctuation is taken into account.

punctuation = ["$,", "$.", "$("]

# PREPROCESSING STARTS HERE
# -o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o

npos_sample_l = int(np.round(sample_length / n_pos))

texts = []
word2vec_models = []

for filename in glob.glob("/Users/user/.../*.txt"):
	fn = filename.split("/")[-1].replace(".txt", "")
	author, title = fn.split("_")
	with open(filename, 'r') as f:
		
		tokens = []
		lemmas = []
		postags = []
		posngrams = []
		function_words = []
		ngrams = []
		to_be_permuted = []
		permutations = []

		# Input file is parsed: tokens, lemmas and PoS-tags assigned to respective empty list 
		for line in f:
			token, lemma, postag = (line.split()[0], line.split()[1], line.split()[2])
			if postag in punctuation:
				pass
			else:
				posngrams.append(postag)
				tokens.append(token)
				lemmas.append(lemma)
				postags.append(postag)
				to_be_permuted.append([token, lemma, postag])
			
		# Four different text representations created from input text (tokens, lemmas, postags and posngrams)
		# Then the texts are sampled into the n_sample_length
		
		tokens = [tokens[i:i+sample_length] for i in range(0, len(tokens), sample_length)]
		lemmas = [lemmas[i:i+sample_length] for i in range(0, len(lemmas), sample_length)]
		postags = [postags[i:i+sample_length] for i in range(0, len(postags), sample_length)]
		to_be_permuted = [to_be_permuted[i:i+sample_length] for i in range(0, len(to_be_permuted), sample_length)]

		# Make n-grams for the parts of speech
		posngrams = [posngrams[i:i+n_pos] for i in range(0, len(posngrams), n_pos)]
		for index, ngram in enumerate(posngrams):
			posngrams[index] = "YY".join(ngram)
		posngrams = [posngrams[i:i+npos_sample_l] for i in range(0, len(posngrams), npos_sample_l)]

		# Per text-sample we take a look at which words we want as our function words
		# We add 'sum' as the only function word as well
		for text_sample in zip(lemmas, postags):
			words = text_sample[0]
			temp_postags = text_sample[1]
			new_list = []
			for index, postag in enumerate(temp_postags):
				if postag in valid_postags:
					new_list.append(words[index])
				if postag == "V" and words[index] == "sum":
					new_list.append(words[index])
			function_words.append(new_list)

		# Make normal n-grams for the tokens
		for text_sample in tokens:
			string = " ".join(text_sample)
			string = list(string)
			for index, char in enumerate(string):
				if char == " ":
					string[index] = "_"
			string = [string[i:i+n] for i in range(0, len(string), n)]
			for index, ngram in enumerate(string):
				string[index] = "".join(ngram)
			ngrams.append(string)

		# Semantic tension differences / Word2vec experiment
		# Make sure there are as many models as there are samples
		for index, text_sample in enumerate(lemmas):
			if len(text_sample) == sample_length:
				text_sample = [text_sample[i:i+15] for i in range(0, len(text_sample), 15)]
				model = gensim.models.Word2Vec(text_sample, min_count=1)
				word2vec_models.append(model)

		# Make to_be_permuted
		for text_sample in to_be_permuted:
			if len(text_sample) == sample_length:
				text_sample = [text_sample[i:i+n_perm] for i in range(0, len(text_sample), n_perm)]
				temp_list = []
				for ngram in text_sample:
					if len(ngram) == n_perm:
						combinations = list(itertools.product(ngram[0], ngram[1], ngram[2]))
						for comb in combinations:
							temp_list.append("YY".join(comb))
				permutations.append(tuple(temp_list))

		# The samples are zipped together and named
		# Those samples which are shorter than the desired sample length are ruled out of the experiment
		samples = zip(tokens, lemmas, postags, posngrams, function_words, ngrams, permutations)
		for index, sample in enumerate(samples):
			if len(sample[0]) == sample_length:
				texts.append((author, title + "_{}".format(str(index+1)), " ".join(sample[0]), 
							  " ".join(sample[1]), " ".join(sample[2]), " ".join(sample[3]), " ".join(sample[4]), 
							  " ".join(sample[5]), " ".join(sample[6])))

authors, titles, tokens, lemmas, postags, posngrams, function_words, ngrams, permutations = zip(*texts)

# VECTORIZING TEXT SAMPLES INTO FREQUENCIES
# -o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o

# Instantiate model for two different kinds of features, one for lemma frequencies, the other for morpho-ngram-frequencies
# The stop_word for v_f_word gives us the liberty to discard some features which we feel to be irrelevant.
# You can delete those irrelevant function words in the invalid_words list in the top of the script
# The TF-IDF vectorizer is a normalization procedure which in our case penalizes often occurring function words in favour of rare function
# words, see Christopher D. Manning et al., Introduction to Information Retrieval, (Cambridge, 2008), 117-33.

v_f_word = TfidfVectorizer(max_features=150, token_pattern=r"(?u)\b\w+\b", stop_words=invalid_words)
v_posngrams = TfidfVectorizer(max_features=150, token_pattern=r"(?u)\b\w+\b")
v_postags = TfidfVectorizer(max_features=10, token_pattern=r"(?u)\b\w+\b")
v_ngrams = TfidfVectorizer(max_features=40, token_pattern=r"(?u)\b\w+\b", stop_words=invalid_ngrams)
v_permutations = TfidfVectorizer(max_features=500, token_pattern=r"(?u)\b\w+\b")

# Make array of counts of MFW words
X_f_word = v_f_word.fit_transform(function_words).toarray()
X_f_word = X_f_word / np.std(X_f_word, axis=0)
vocab_f_word = v_f_word.get_feature_names()

# Make array of n_pos-grams of PoS-tags
X_posngrams = v_posngrams.fit_transform(posngrams).toarray()
X_posngrams = X_posngrams / np.std(X_posngrams, axis=0)
vocab_posngrams = v_posngrams.get_feature_names()

# Make array of ngrams for tokens
X_ngrams = v_ngrams.fit_transform(ngrams).toarray()
X_ngrams = X_ngrams / np.std(X_ngrams, axis=0)
vocab_ngrams = v_ngrams.get_feature_names()

# Make array of postag frequencies
X_postags = v_postags.fit_transform(postags).toarray()
X_postags = X_postags / np.std(X_postags, axis=0)
vocab_pos = v_postags.get_feature_names()

# Make array of permutation frequencies
X_permutations = v_permutations.fit_transform(permutations).toarray()
X_permutations = X_permutations / np.std(X_permutations, axis=0)
vocab_perm = v_permutations.get_feature_names()

# WORD2VEC EXPERIMENT
# -o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o

# Huge list of features based on the semantics of our function words
# This was an experiment unmentioned in the article, it can be left out of the code.

long_list = []
for author, title, model in zip(authors, titles, word2vec_models):
	f_word_vec = np.array([])
	for f_word in vocab_f_word:
		try:
			f_word_vec = np.append(f_word_vec, model[f_word])
		except KeyError:
			f_word_vec = np.append(f_word_vec, np.zeros(100))
	long_list.append(f_word_vec)
X_w2v = np.matrix([vec for vec in long_list])
X_w2v = X_w2v / np.std(X_w2v, axis=0)

# -o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o

# Make array for X, in which the desired features are taken into account
# Add up all the features in one list

features = vocab_f_word
X = X_f_word

# If desired, you can concatenate multiple features into one array, unhash this line of code:

#X = np.concatenate((X_posngrams, X_permutations, X_f_word), axis=1)

# CREATING DISTANCE DATA FOR GEPHI, COMPUTING K NEAREST NEIGHBOURS BY FUNCTION WORD FREQUENCIES
# -o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o

# Here we make the tab-separated txt files which should go into GEPHI; we instantiate a file object.
# Calculates Nearest Neighbours of each txt sample.

fob_nodes = open("/Users/user/...", "w")
fob_edges = open("/Users/user/...", "w")
fob_nodes.write("Id" + "\t" + "Work" + "\t" + "Author" + "\n")
fob_edges.write("Source" + "\t" + "Target" + "\t" + "Type" + "\t" + "Weight" + "\n")

# Weights in rank of closest distance and k:

weights =  [1.0, 0.95, 0.9, 0.8, 0.7]
k = 6
nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X_f_word)
distances, indices = nbrs.kneighbors(X)

# Group our nearest texts:

nearest_texts = []
for distance, index_list in zip(distances, indices):
	nearest_texts.append(((titles[index_list[1]], authors[index_list[1]], weights[0], index_list[1] + 1), 
						(titles[index_list[2]], authors[index_list[2]], weights[1], index_list[2] + 1), 
						(titles[index_list[3]], authors[index_list[3]], weights[2], index_list[3] + 1),
						(titles[index_list[4]], authors[index_list[4]], weights[3], index_list[4] + 1),
						(titles[index_list[5]], authors[index_list[5]], weights[4], index_list[5] + 1),))

# Write out to files:

for index, (author, title, nearest_text) in enumerate(zip(authors, titles, nearest_texts)):
	fob_nodes.write(str(index + 1) + "\t" + str(title) + "\t" + str(author) + "\n")
	fob_edges.write(str(index+1) + "\t" + str(nearest_text[0][3]) + "\t" + "Undirected" + "\t" + str(nearest_text[0][2]) + "\n" +
					str(index+1) + "\t" + str(nearest_text[1][3]) + "\t" + "Undirected" + "\t" + str(nearest_text[1][2]) + "\n" +
					str(index+1) + "\t" + str(nearest_text[2][3]) + "\t" + "Undirected" + "\t" + str(nearest_text[2][2]) + "\n" +
					str(index+1) + "\t" + str(nearest_text[3][3]) + "\t" + "Undirected" + "\t" + str(nearest_text[3][2]) + "\n" +
					str(index+1) + "\t" + str(nearest_text[4][3]) + "\t" + "Undirected" + "\t" + str(nearest_text[4][2]) + "\n")

# WORKING ON PLOTS AND PCA
# -o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o

# Instantiate colours for the plot on the PCA (authors)
# b: blue, g: green, r: red, c: cyan, m: magenta, y: yellow, k: black, w: white
# You can also use #hex color strings as '#988ED5'

colours = {'brev': '#2a7b9c', 'new': '#64b9db', 'pre': '#e47c66', 'post': '#ffb2a2', 'mid': '#ff8b72', 'nic': '#add45c', 'sc': '#2a7b9c', 
		   'div': '#c6c8c5', 'dub': '#e47c66', 'hugo': '#00baae'}

# Plot PCA functions with our new data points which we received in X_bar (pca.fit_transform(X_input)).
# Inside the PCA plots we instantiate the number of components and reduce X to the size of required features.
# We get a score of how much of the variance the plot explains.

def PCA_2D(X_input):

	x = input("Show loadings? (y/n) ")

	# Initialize PCA
	pca = PCA(n_components=2)
	X_bar = pca.fit_transform(X_input)
	var_exp = pca.explained_variance_ratio_
	comps = pca.components_
	comps = comps.transpose()
	loadings = pca.components_.transpose()
	vocab_weights_p1 = sorted(zip(features, comps[:,0]), key=lambda tup: tup[1], reverse=True)
	vocab_weights_p2 = sorted(zip(features, comps[:,1]), key=lambda tup: tup[1], reverse=True)

	# Plot PCA with loadings
	if x == "y":

		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		x1, x2 = X_bar[:,0], X_bar[:,1]
		ax1.scatter(x1, x2, edgecolors='none', facecolors='none')
		for p1, p2, a, title in zip(x1, x2, authors, titles):
			ax1.text(p1, p2, a[:3] + '_' + title.split("_")[1], ha='center',
				    va='center', color=colours[a])
		ax1.set_xlabel('PC1')
		ax1.set_ylabel('PC2')

		ax2 = ax1.twinx().twiny()
		l1, l2 = loadings[:,0], loadings[:,1]
		ax2.scatter(l1, l2, 100, edgecolors='none', facecolors='none')
		for x, y, l in zip(l1, l2, features):
			ax2.text(x, y, l, ha='center', va='center', color='darkgrey',
				fontdict={'family': 'Arial', 'size': 12})

		plt.show()

	elif x == "n":
		fig = plt.figure()
		ax = fig.add_subplot(111)
		x1, x2 = X_bar[:,0], X_bar[:,1]

		ax.scatter(x1, x2, 100, edgecolors='none', facecolors='none')
		for p1, p2, a, title in zip(x1, x2, authors, titles):
			ax.text(p1, p2, title[:3] + '_' + title.split("_")[1], ha='center',
			        va='center', color=colours[a], fontdict={'size': 13})
		ax.set_xlabel('PC1')
		ax.set_ylabel('PC2')

		# Make a legend:

		brev_patch = mpatches.Patch(color=colours['brev'], label='Bernard\'s intra corpus (brevis)')
		new_patch = mpatches.Patch(color=colours['new'], label='Bernard\'s intra corpus (perfectum additions)')
		pre_patch = mpatches.Patch(color=colours['pre'], label='Bernard\'s extra corpus (pre-1140)')
		mid_patch = mpatches.Patch(color=colours['mid'], label='Bernard\'s extra corpus (1140-1145)')
		post_patch = mpatches.Patch(color=colours['post'], label='Bernard\'s extra corpus (post-1145')
		nic_patch = mpatches.Patch(color=colours['nic'], label='Nicholas\' letters and sermons')

		plt.legend(handles=[brev_patch, new_patch, pre_patch, mid_patch, post_patch, nic_patch], loc=1, prop={'size':9})

		#plt.plot([0,0], [1, 2], ":", lw=2)
		plt.axhline(y=0, ls="--", lw=1.5, c='0.75')
		plt.axvline(x=0, ls="--", lw=1.5, c='0.75')
		
		plt.show()
		fig.savefig("/Users/jedgusse/Stylo_R/fig.pdf", transparent=True, format='pdf')
		plt.close()


def PCA_3D(X_input):

	# Initialize PCA
	pca = PCA(n_components=3)
	X_bar = pca.fit_transform(X_input)
	var_exp = pca.explained_variance_ratio_
	comps = pca.components_

	print("Variance explained: ", var_exp[0] + var_exp[1] + var_exp[2])
	print("Word types taken into account: ", ", ".join(valid_postags))
	print("Features used: ", len(features), ":", ", ".join(features))

	# Plot PCA
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x1, x2, x3 = X_bar[:,0], X_bar[:,1], X_bar[:,2]
	ax.scatter(x1, x2, x3, edgecolors='none', facecolors='none')
	for p1, p2, p3, a, title in zip(x1, x2, x3, authors, titles):
		ax.text(p1, p2, p3, a[:3] + '_' + title, ha='center',
				va='center', color=colours[a])
	ax.set_xlabel('PC1')
	ax.set_ylabel('PC2')
	ax.set_zlabel('PC3')

	plt.show()


def plot_features(X_input):

	# Initialize PCA
	pca = PCA(n_components=2)
	X_bar = pca.fit_transform(X_input)
	var_exp = pca.explained_variance_ratio_
	comps = pca.components_
	comps = comps.transpose()

	fig = plt.figure(figsize=(10,10))
	ax1 = fig.add_subplot(111)
	l1, l2 = comps[:,0], comps[:,1]
	ax1.scatter(l1, l2, 100, edgecolors='none', facecolors='none')
	for x, y, l in zip(l1, l2, features):
		ax1.text(x, y, l, ha='center', va='center', color='darkgrey',
				 fontdict={'family': 'Arial', 'size': 12})

	plt.show()
