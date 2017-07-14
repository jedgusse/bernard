#!/usr/bin/env

import argparse
from binascii import hexlify
from collections import Counter
import colorsys
import itertools
from itertools import combinations
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
from nltk import ngrams
import numpy as np
import random
import seaborn.apionly as sns
from scipy import stats
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.preprocessing import StandardScaler
from string import punctuation
import sys
import pandas as pd
from preprocess import DataReader, Vectorizer

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def align_yaxis(ax1, v1, ax2, v2):
	"""adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
	_, y1 = ax1.transData.transform((0, v1))
	_, y2 = ax2.transData.transform((0, v2))
	inv = ax2.transData.inverted()
	_, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
	miny, maxy = ax2.get_ylim()
	ax2.set_ylim(miny+dy, maxy+dy)

def align_xaxis(ax1, v1, ax2, v2):
	"""adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
	x1, _ = ax1.transData.transform((v1, 0))
	x2, _ = ax2.transData.transform((v2, 0))
	inv = ax2.transData.inverted()
	dx, _ = inv.transform((0, 0)) - inv.transform((x1-x2, 0))
	minx, maxx = ax2.get_xlim()
	ax2.set_xlim(minx+dx, maxx+dx)

class PrinCompAnal:

	""" |--- Principal Components Analysis ---|
		::: Plots PCA Plot ::: """

	def __init__(self, authors, titles, X, features, sample_size, n_components):
		self.authors = authors
		self.titles = titles
		self.X = X
		self.features = features
		self.sample_size = sample_size
		self.n_components = n_components

	def plot(self, show_samples, show_loadings, sbrn_plt):

		# Normalizer and Delta perform badly
		# They flatten out all difference in a PCA plot
		
		pca = PCA(n_components=self.n_components)
		X_bar = pca.fit_transform(self.X)
		var_exp = pca.explained_variance_ratio_
		var_pc1 = np.round(var_exp[0]*100, decimals=2)
		var_pc2 = np.round(var_exp[1]*100, decimals=2)
		explained_variance = np.round(sum(pca.explained_variance_ratio_)*100, decimals=2)
		comps = pca.components_
		comps = comps.transpose()
		loadings = pca.components_.transpose()
		vocab_weights_p1 = sorted(zip(self.features, comps[:,0]), key=lambda tup: tup[1], reverse=True)
		vocab_weights_p2 = sorted(zip(self.features, comps[:,1]), key=lambda tup: tup[1], reverse=True)

		if sbrn_plt == False:

			# Generate color dictionary
			color_dict = {author:index for index, author in enumerate(sorted(set(self.authors)))}
			cmap = discrete_cmap(len(color_dict), base_cmap='brg')

			if show_samples == True:

				fig = plt.figure(figsize=(8,6))
				ax = fig.add_subplot(111)
				x1, x2 = X_bar[:,0], X_bar[:,1]

				# If anything needs to be invisible in plot, add to exclusion_list

				exclusion_list = []

				potential_dict = {'sciv': '#D8E665', 'ldo': '#e67465', 'lvm': '#6597e6'}
				potential_dict2 = {'sciv': 'Scivias', 'ldo': 'Liber diuinorum operum', 'lvm': 'Liber uite meritorum'}
				ax.scatter(x1, x2, 100, edgecolors='none', facecolors='none', cmap='rainbow')
				for index, (p1, p2, a, title) in enumerate(zip(x1, x2, self.authors, self.titles)):
					# ax.scatter(p1, p2, marker='o', color=cmap(color_dict[a]), s=20)
					ax.scatter(p1, p2, marker='o', color=potential_dict[a], s=20)
					ax.text(p1, p2, title.split('_')[-1], color='black', fontdict={'size': 5})
					# if a not in exclusion_list:
					# 	ax.text(p1, p2, title, ha='center',
					#     va='center', color=cmap(color_dict[a]), 
					#     fontdict={'size': 7})
					

				"""||| LEGEND FOR ARTICLE ENRIQUE AND MIKE ||| """		
				"""ax.scatter(x1, x2, 100, edgecolors='none', facecolors='none', cmap='rainbow')
				for index, (p1, p2, a, title) in enumerate(zip(x1, x2, self.authors, self.titles)):
					if title.split('-')[0] == 'gener':
						ax.scatter(p1, p2, marker='^', color=cmap(color_dict[a]), s=9)
					else:
						ax.scatter(p1, p2, marker='o', color=cmap(color_dict[a]), s=9)"""

				# Legend settings (code for making a legend)

				collected_patches = []
				for author in set(self.authors):
					# legend_patch = mpatches.Patch(color=cmap(color_dict[author]), label=author.split('-')[0])
					legend_patch = mpatches.Patch(color=potential_dict[author], label=potential_dict2[author])
					collected_patches.append(legend_patch)

				#generated_triangle = ax.scatter([],[], marker='^', color='black', label=r'Ngram-Generated $\bar{\alpha}$')
				#real_circle = ax.scatter([],[], marker='o', color='black', label=r'Authentic $\omega$')
				#collected_patches.append(generated_triangle)
				#collected_patches.append(real_circle)
				plt.legend(handles=collected_patches, fontsize=7)

				ax.set_xlabel('Principal Component 1 \n \n Explained Variance: {}% \n Sample Size: {} words/sample \n Number of Features: {} features'.format(str(explained_variance), str(self.sample_size), str(len(self.features))), fontdict={'size': 7})
				ax.set_ylabel('Principal Component 2', fontdict={'size': 7})

				#ax.set_xlabel('Principal Component 1: {}%'.format(var_pc1))
				#ax.set_ylabel('Principal Component 2: {}%'.format(var_pc2))

				if show_loadings == True:
					ax2 = ax.twinx().twiny()
					l1, l2 = loadings[:,0], loadings[:,1]
					ax2.scatter(l1, l2, 100, edgecolors='none', facecolors='none');
					for x, y, l in zip(l1, l2, self.features):
						ax2.text(x, y, l, ha='center', va="center", color="black",
						fontdict={'family': 'Arial', 'size': 6})

					# Align axes

					# Important to adjust margins first when function words fall outside plot
					# This is due to the axes aligning (def align).

					ax2.margins(x=0.14, y=0.14)

					align_xaxis(ax, 0, ax2, 0)
					align_yaxis(ax, 0, ax2, 0)

					plt.axhline(y=0, ls="--", lw=0.5, c='0.75')
					plt.axvline(x=0, ls="--", lw=0.5, c='0.75')
					
					plt.tight_layout()
					plt.show()
				
				elif show_loadings == False:

					plt.axhline(y=0, ls="--", lw=0.5, c='0.75')
					plt.axvline(x=0, ls="--", lw=0.5, c='0.75')

					plt.tight_layout()
					plt.show()

					# Converting PDF to PNG, use pdftoppm in terminal and -rx -ry for resolution settings

				fig.savefig("/Users/jedgusse/compstyl/output/fig_output/pcafig.pdf", transparent=True, format='pdf')

			elif show_samples == False:

				fig = plt.figure(figsize=(8, 6))
				ax2 = fig.add_subplot(111)
				l1, l2 = loadings[:,0], loadings[:,1]
				ax2.scatter(l1, l2, 100, edgecolors='none', facecolors='none')
				for x, y, l in zip(l1, l2, features):
					ax2.text(x, y, l, ha='center', va='center', color='black',
						fontdict={'family': 'Arial', 'size': 6})

				ax2.set_xlabel('PC1')
				ax2.set_ylabel('PC2')

				align_xaxis(ax, 0, ax2, 0)
				align_yaxis(ax, 0, ax2, 0)

				plt.axhline(y=0, ls="--", lw=0.5, c='0.75')
				plt.axvline(x=0, ls="--", lw=0.5, c='0.75')

				plt.tight_layout()
				plt.show()
				fig.savefig("/Users/jedgusse/compstyl/output/fig_output/pcafig.pdf", bbox_inches='tight', transparent=True, format='pdf')

				# Converting PDF to PNG, use pdftoppm in terminal and -rx -ry for resolution settings

		else:

			data = [(title.split("_")[0], author, pc1, pc2) for [pc1, pc2], title, author in zip(X_bar, self.titles, self.authors)]
			df = pd.DataFrame(data, columns=['title', 'author', 'PC1', 'PC2'])

			# Get the x in an array
			sns.set_style('darkgrid')
			sns_plot = sns.lmplot('PC1', 'PC2', data=df, fit_reg=False, hue="author",
			           scatter_kws={"marker": "+","s": 100}, markers='o', legend=False)

			plt.legend(loc='upper right')
			plt.tight_layout()
			plt.show()

			sns_plot.savefig("/Users/jedgusse/compstyl/output/fig_output/pcasbrn.pdf")
			
class GephiNetworks:

	""" |--- Gephi Networks (k-NN Networks) ---|
		::: Yields k-Nearest Neighbor Network ::: """

	def __init__(self, folder_location, sample_size, invalid_words):
		self.folder_location = folder_location
		self.sample_size = sample_size
		self.invalid_words = invalid_words

	def plot(self, feat_range, random_sampling, corpus_size):

		# This is the standard number of neighbors. This cannot change unless the code changes.
		n_nbrs = 4

		# 3 neighbors for each sample is argued to make up enough consensus
		# Try to make a consensus of distance measures
		# Use cosine, euclidean and manhattan distance, and make consensus tree (inspired by Eder)
		# Also search over ranges of features to make the visualization less biased

		metric_dictionary = {'manhattan': 'manhattan', 'cosine': 'cosine', 'euclidean': 'euclidean'}

		authors, titles, texts = DataReader(self.folder_location, self.sample_size, {}, {}
										).metadata(sampling=True,
										type='folder',
										randomization=False)

		# random Stratified Sampling 
		# each sample receives its sampling fraction corresponding to proportionate number of samples

		corpus_size = corpus_size*1000

		if random_sampling == 'stratified':
			strata_proportions = {title.split('_')[0]: np.int(np.round(int(title.split('_')[-1]) / len(titles) * corpus_size / self.sample_size)) for title in titles}
			# print('::: corpus is being stratified to {} words in following proportions : '.format(str(corpus_size)))
			# print(strata_proportions, ' :::')
			strat_titles = []
			for stratum in strata_proportions:
				strata = [title for title in titles if stratum == title.split('_')[0]]
				sampling_fraction = strata_proportions[stratum]
				local_rand_strat_titles = random.sample(strata, sampling_fraction)
				strat_titles.append(local_rand_strat_titles)
			strat_titles = sum(strat_titles, [])
			strat_authors = [author for author, title in zip(authors, titles) if title in strat_titles]
			strat_texts = [text for title, text in zip(titles, texts) if title in strat_titles]
			titles = strat_titles
			authors = strat_authors
			texts = strat_texts

		fob_nodes = open("/Users/jedgusse/compstyl/output/gephi_output/gephi_nodes.txt", "w")
		fob_edges = open("/Users/jedgusse/compstyl/output/gephi_output/gephi_edges.txt", "w")

		fob_nodes.write("Id" + "\t" + "Work" + "\t" + "Author" + "\n")
		fob_edges.write("Source" + "\t" + "Target" + "\t" + "Type" + "\t" + "Weight" + "\n")

		# Build up consensus distances of different feature ranges and different metrics
		exhsearch_data = []
		for n_feats in feat_range:
			# print("::: running through feature range {} ::: ".format(str(n_feats)))
			tfidf_vectors, tfidf_features = Vectorizer(texts, self.invalid_words,
										  n_feats=n_feats,
										  feat_scaling='standard_scaler',
										  analyzer='word',
										  vocab=None
										  ).tfidf(smoothing=True)
			if n_feats == feat_range[-1]:
				pass
				# print("FEATURES: ", ", ".join(tfidf_features))
			for metric in metric_dictionary:
				model = NearestNeighbors(n_neighbors=n_nbrs,
										algorithm='brute',
										metric=metric_dictionary[metric],
										).fit(tfidf_vectors)
				distances, indices = model.kneighbors(tfidf_vectors)
				
				# Distances are normalized in order for valid ground for comparison
				all_distances = []
				for distance_vector in distances:
					for value in distance_vector:
						if value != 0.0:
							all_distances.append(value)

				all_distances = np.array(all_distances)
				highest_value = all_distances[np.argmin(all_distances)]
				lowest_value = all_distances[np.argmax(all_distances)]
				normalized_distances = (distances - lowest_value) / (highest_value - lowest_value)
				
				# Distances appended to dataframe
				for distance_vec, index_vec in zip(normalized_distances, indices):
					data_tup = ('{} feats, {}'.format(str(n_feats), metric_dictionary[metric]),
								titles[index_vec[0]], 
								titles[index_vec[1]], distance_vec[1],
								titles[index_vec[2]], distance_vec[2],
								titles[index_vec[3]], distance_vec[3])
					exhsearch_data.append(data_tup)

		# Entire collected dataframe
		df = pd.DataFrame(exhsearch_data, columns=['exp', 'node', 'neighbor 1', 'dst 1', 'neighbor 2', 
										 'dst 2', 'neighbor 3', 'dst 3']).sort_values(by='node', ascending=0)
		final_data = []
		weights= []
		node_orientation  = {title: idx+1 for idx, title in enumerate(titles)}
		for idx, (author, title) in enumerate(zip(authors, titles)):
			neighbors = []
			dsts = []
			# Pool all neighbors and distances together (ignore ranking of nb1, nb2, etc.)
			for num in range(1, n_nbrs):
				neighbors.append([neighb for neighb in df[df['node']==title]['neighbor {}'.format(str(num))]])
				dsts.append([neighb for neighb in df[df['node']==title]['dst {}'.format(str(num))]])
			neighbors = sum(neighbors, [])
			dsts = sum(dsts, [])

			model = CountVectorizer(lowercase=False)
			count_dict = model.fit_transform(neighbors)
			
			# Collect all the candidates per sample that were chosen by the algorithm as nearest neighbor at least once
			candidate_dict = {neighbor: [] for neighbor in model.get_feature_names()}
			for nbr, dst in zip(neighbors, dsts):
				candidate_dict[nbr].append(dst)
			candidate_dict = {nbr: np.mean(candidate_dict[nbr])*len(candidate_dict[nbr]) for nbr in candidate_dict}
			candidate_dict = sorted(candidate_dict.items(), key=lambda x: x[1], reverse=True)

			fob_nodes.write(str(idx + 1) + "\t" + str(title.split('_')[-1]) + "\t" + str(author) + "\n")
			data_tup = (title,)
			for candtitle, weight in candidate_dict[:8]:
				data_tup = data_tup + (candtitle, weight,)
				weights.append(weight)
				fob_edges.write(str(idx+1) + "\t" + str(node_orientation[candtitle]) + "\t" + "Undirected" + "\t" + str(weight) + "\n")
			final_data.append(data_tup)

		# Prepare column names for dataframe
		longest = np.int((len(final_data[np.argmax([len(i) for i in final_data])]) - 1) / 2)
		columns = sum([['neighbor {}'.format(str(i)), 'dst {}'.format(str(i))] for i in range(1, longest+1)], [])
		columns.insert(0, 'node')
		final_df = pd.DataFrame(final_data, columns=columns).sort_values(by='node', ascending=0)

		# Results
		# print('::: RESULTS :::')
		# print(final_df.head())
		# print('::: VARIANCE BETWEEN DISTANCES :::')
		return np.var(np.array(weights))

class VoronoiDiagram:

	""" |--- Gives realistic estimate of 2D-Decision boundary ---|
		::: Only takes grid_doc_vectors ::: """

	def __init__(self):
		self.grid_doc_vectors = grid_doc_vectors
		self.Y_train = Y_train
		self.y_predicted = y_predicted
		self.best_n_feats = best_n_feats
		self.ordered_authors = ordered_authors
		self.ordered_titles = ordered_titles

	def plot(self):
		
		colours = {'Bern': '#000000', 'NicCl': '#000000', 'lec': '#ff0000', 
				   'ro': '#ff0000', 'Alain': '#000000', 'AnsLaon': '#000000', 
			   	   'EberYpr': '#000000', 'Geof': '#000000', 'GilPoit': '#000000'}

		# Plotting SVM

		# Dimensions of the data reduced in 2 steps - from 300 to 50, then from 50 to 2 (this is a strong recommendation).
		# t-SNE (t-Distributed Stochastic Neighbor Embedding): t-SNE is a tool for data visualization. 
		# Local similarities are preserved by this embedding. 
		# t-SNE converts distances between data in the original space to probabilities.
		# In contrast to, e.g., PCA, t-SNE has a non-convex objective function. The objective function is minimized using a gradient descent 
		# optimization that is initiated randomly. As a result, it is possible that different runs give you different solutions.

		# First, reach back to original values, and put in the new y predictions in order to draw up the Voronoi diagram, which is basically
		# a 1-Nearest Neighbour fitting. (For the Voronoi representation, see MLA	
	    # Migut, M. A., Marcel Worring, and Cor J. Veenman. "Visualizing multi-dimensional decision boundaries in 2D." 
	    # Data Mining and Knowledge Discovery 29.1 (2015): 273-295.)

		# IMPORTANT: we take the grid_doc_vectors (not original data), those feature vectors which the SVM itself has made the decision on.
		# We extend the y vector with the predicted material

		print('::: running t-SNE for dimensionality reduction :::')

		y = np.append(self.Y_train, self.y_predicted, axis=0)

		# If features still too many, truncate the grid_doc_vectors to reasonable amount, then optimize further
		# A larger / denser dataset requires a larger perplexity

		if self.best_n_feats < 50:
			X_embedded = TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(self.grid_doc_vectors)
		else:
			X_reduced = TruncatedSVD(n_components=50, random_state=0).fit_transform(self.grid_doc_vectors)
			X_embedded = TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(X_reduced)

		# create meshgrid
		resolution = 100 # 100x100 background pixels
		X2d_xmin, X2d_xmax = np.min(X_embedded[:,0]), np.max(X_embedded[:,0])
		X2d_ymin, X2d_ymax = np.min(X_embedded[:,1]), np.max(X_embedded[:,1])
		xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, resolution), np.linspace(X2d_ymin, X2d_ymax, resolution))

		# Approximate Voronoi tesselation on resolution x resolution grid using 1-NN
		background_model = KNeighborsClassifier(n_neighbors=1).fit(X_embedded, y)
		voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
		voronoiBackground = voronoiBackground.reshape((resolution, resolution))

		# (http://stackoverflow.com/questions/37718347/plotting-decision-boundary-for-high-dimension-data)
		vor = Voronoi(X_embedded)

		fig = plt.figure(figsize=(10,8))

		# Define colour mapping

		plt.contourf(xx, yy, voronoiBackground, levels=[0, 0.5, 1], colors=('#eaeaea', '#b4b4b4'))
		ax = fig.add_subplot(111)
		
		ax.scatter(X_embedded[:,0], X_embedded[:,1], 100, edgecolors='none', facecolors='none')
		for p1, p2, a, title in zip(X_embedded[:,0], X_embedded[:,1], self.ordered_authors, self.ordered_titles):
			ax.text(p1, p2, title[:2] + '_' + title.split("_")[1], ha='center',
		    va='center', color=colours[a], fontdict={'size': 7})
		for vpair in vor.ridge_vertices:
		    if vpair[0] >= 0 and vpair[1] >= 0:
		    	v0 = vor.vertices[vpair[0]]
		    	v1 = vor.vertices[vpair[1]]
		    	# Draw a line from v0 to v1.
		    	plt.plot([v0[0], v1[0]], [v0[1], v1[1]], 'k', linewidth=0.3, linestyle='--')

		ax.set_xlabel('F1 (Conditional probability)')
		ax.set_ylabel('F2 (Conditional probability)')

		plt.axis([X2d_xmin, X2d_xmax, X2d_ymin, X2d_ymax])

		plt.show()

		fig.savefig("/Users/jedgusse/stylofactory/output/fig_output/voronoi_fig.pdf", transparent=True, format='pdf')

class RollingDelta:

	""" |--- Rolling Delta ---|
		::: Roll test vectors over centroid train vector ::: """

	def __init__(self, folder_location, n_feats, invalid_words, sample_size, step_size, test_dict, rnd_dct):
		self.folder_location = folder_location
		self.n_feats = n_feats
		self.invalid_words = invalid_words
		self.sample_size = sample_size
		self.step_size = step_size
		self.test_dict = test_dict
		self.rnd_dct = rnd_dct

	def plot(self):

		# Make a train_test_split
		# The training corpus is the benchmark corpus

		train_data = []
		train_metadata = []
		
		test_data = []
		test_metadata = []

		# Make a split by using the predefined test_dictionary

		print("::: test - train - split :::")

		for filename in glob.glob(self.folder_location + '/*'):
			author = filename.split("/")[-1].split(".")[0].split("_")[0]
			title = filename.split("/")[-1].split(".")[0].split("_")[1]

			if title not in self.test_dict.values():
				author, title, text = DataReader(filename, 
										self.sample_size, self.test_dict,
										self.rnd_dct).metadata(sampling=True,
										type='file', randomization=False)
				train_metadata.append((author, title))
				train_data.append(text)

			elif title in self.test_dict.values():
				author, title, text = DataReader(filename, 
										self.sample_size, self.test_dict,
										self.rnd_dct).metadata(sampling=False, 
										type='file', randomization=False)
				test_metadata.append((author, title))
				test_data.append(text.split())

		# Unnest nested list
		# Preparing the two corpora for take-off
		train_data = sum(train_data, [])

		print("::: vectorizing training corpus :::")

		# Vectorize training data
		doc_vectors, features = Vectorizer(train_data, self.invalid_words,
									  n_feats=self.n_feats,
									  feat_scaling=False,
									  analyzer='word',
									  vocab=None
									  ).raw()

		# We first turn our raw counts into relative frequencies
		relative_vectors = [vector / np.sum(vector) for vector in doc_vectors]
		
		# We produce a standard deviation vector, that will later serve to give more weight to highly changeable words and serves to
		# boost words that have a low frequency. This is a normal Delta procedure.
		# We only calculate the standard deviation on the benchmark corpus, since that is the distribution against which we want to compare
		stdev_vector = np.std(relative_vectors, axis = 0)

		# We make a centroid vector for the benchmark corpus
		centroid_vector = np.mean(relative_vectors, axis=0)

		# Now we have turned all the raw counts of the benchmark corpus into relative frequencies, and there is a centroid vector
		# which counts as a standard against which the test corpus can be compared.

		# We now divide the individual test texts in the given sample lengths, taking into account the step_size of overlap
		# This is the "shingling" procedure, where we get overlap, where we get windows	

		# Get highest x value
		lengths = np.array([len(text) for text in test_data])
		maxx = lengths[np.argmax(lengths)]

		print("::: making step-sized windows and rolling out test data :::")

		all_data = []
		for (author, title), test_text in zip(test_metadata, test_data):

			steps = np.arange(0, len(test_text), self.step_size)
			step_ranges = []

			windowed_samples = []
			for each_begin in steps:
				sample_range = range(each_begin, each_begin + self.sample_size)
				step_ranges.append(sample_range)
				text_sample = []
				for index, word in enumerate(test_text):
					if index in sample_range:
						text_sample.append(word)
				windowed_samples.append(text_sample)

			# Now we change the samples to numerical values, using the features as determined in code above
			# Only allow text samples that have desired sample length

			window_vectors = []
			for text_sample in windowed_samples:
				if len(text_sample) == self.sample_size:
					vector = []
					counter = Counter(text_sample)
					for feature in features:
						vector.append(counter[feature])
					window_vectors.append(vector)
			window_vectors = np.asarray(window_vectors)

			window_relative = [vector / np.sum(vector) for vector in window_vectors]

			delta_scores = []
			for vector in window_relative:
				delta_distances = np.mean(np.absolute(centroid_vector - vector) / stdev_vector)
				delta_score = np.mean(delta_distances)
				delta_scores.append(delta_score)

			x_values = [graphthing[-1] for graphthing, sample in zip(step_ranges, windowed_samples) if len(sample) == self.sample_size]

			data = [(author, title, x+1, y) for x, y in zip(x_values, delta_scores)]
			all_data.append(data)

		all_data = sum(all_data, [])
		df = pd.DataFrame(all_data, columns=['author', 'title', 'x-value', 'delta-value'])

		# Plot with seaborn

		fig = plt.figure(figsize=(20,5))

		sns.plt.title('Rolling Delta')
		sns.set(font_scale=0.5)
		sns.set_style("whitegrid")

		ax = sns.pointplot(x=df['x-value'], y=df['delta-value'], data=df, ci=5, scale=0.4, hue='author')

		ax.set_xlabel("Step Size: {} words, Sample Size: {} words".format(str(self.step_size), str(self.sample_size)))
		ax.set_ylabel("Delta Score vs. Centroid Vector")

		# Set good x tick labels
		for ind, label in enumerate(ax.get_xticklabels()):
		    if ind % 30 == 0:
		    	label.set_visible(True)
		    else:
		    	label.set_visible(False)

		sns.plt.show()
		fig.savefig("/Users/jedgusse/compstyl/output/fig_output/rollingdelta.pdf", bbox_inches='tight')

class IntrinsicPlagiarism:

	""" |--- Intrinsic Plagiarism ---|
		::: N-gram profiles and a sliding window with no reference corpus ::: """

	def __init__(self, folder_location, n_feats, invalid_words, sample_size, step_size):
		self.folder_location = folder_location
		self.n_feats = n_feats
		self.invalid_words = invalid_words
		self.sample_size = sample_size
		self.step_size = step_size

	def plot(self, support_ngrams, support_punct):
		
		# Make sure the analyzer is on the character or word level.
		analyzer = ''
		n = 3
		ngram_range = None
		if support_ngrams == False:
			analyzer += 'word'
			ngram_range = ((1,1))
		else:
			analyzer += 'char'
			# Advised by Stamatatos: the 3gram range
			ngram_range = ((n,n))

		# Open file and set up stepsized samples
		filename = glob.glob(self.folder_location+"/*")
		if len(filename) > 1:
			sys.exit("-- | ERROR: Intrinsic plagiarism detection can handle only 1 file")
		fob = open(filename[0])
		text = fob.read()
		bulk = []
		if support_punct == False:
			for feat in text.strip().split():
				feat = "".join([char for char in feat if char not in punctuation])
				bulk.append(feat)

		if analyzer == 'word':
			text = bulk
		elif analyzer == 'char':
			text = " ".join(bulk)

		# Make sure the texts are split when words are analyzed
		# Also the reconnection of features in the texts happens differently
		print("::: creating sliding windows :::")
		steps = np.arange(0, len(text), self.step_size)
		step_ranges = []
		windowed_samples = []
		for each_begin in steps:
			sample_range = range(each_begin, each_begin + self.sample_size)
			step_ranges.append(sample_range)
			text_sample = []
			for index, feat in enumerate(text):
				if index in sample_range:
					text_sample.append(feat)
			if len(text_sample) == self.sample_size:
				if analyzer == 'char':
					windowed_samples.append("".join(text_sample))
				elif analyzer == 'word':
					windowed_samples.append(" ".join(text_sample))
		if analyzer == 'word':
			text = " ".join(text)

		print("::: converting windows to document vectors :::")
		model = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range, 
								stop_words=self.invalid_words)
		doc_vector = model.fit_transform([text]).toarray().flatten()
		doc_vector = doc_vector / len(text)
		vocab = model.get_feature_names()

		print("::: calculating dissimilarity measures :::")
		# Count with predefined vocabulary based on entire document
		ds = []
		model = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range, 
								vocabulary=vocab, stop_words=self.invalid_words)
		for sample in windowed_samples:
			sample_vector = model.fit_transform([sample]).toarray().flatten()
			sample_vector = sample_vector / len(sample)
			dissimilarity_measure = np.power(np.mean(np.divide(2*(sample_vector - doc_vector), sample_vector + doc_vector)), 2)
			ds.append(dissimilarity_measure)
		
		# Set up threshold; in calculating the threshold, ignore likely-to-be plagiarized passages 
		filter = np.mean(ds) + np.std(ds)
		averaged_ds = [i for i in ds if i <= filter]
		filter_threshold = np.mean(averaged_ds) + 2*np.std(averaged_ds)

		print("::: visualizing style change function «sc» :::")
		if analyzer == 'char':
			x_values = [graphthing[-1] for graphthing, sample in zip(step_ranges, windowed_samples) if len(sample) == self.sample_size]
		elif analyzer == 'word':
			x_values = [graphthing[-1] for graphthing, sample in zip(step_ranges, windowed_samples)]
		data = [(x+1, y, filter_threshold) for x, y in zip(x_values, ds)]
		df = pd.DataFrame(data, columns=['range', 'dissimilarity measure', 'filter_threshold'])

		# Exporting plagiarized text to database
		print("::: ranges and detected stylistic outliers :::")
		df_plag = []
		for s_range, dissimilarity_measure in zip(step_ranges, ds):
			if dissimilarity_measure >= filter_threshold:
				range_string = str(s_range[0]) + "-" + str(s_range[-1])
				plag_text = "".join(text[index] for index in s_range)
				df_plag.append((range_string, plag_text))
		df_plag = pd.DataFrame(df_plag, columns=['{} range'.format(analyzer), 'plagiarized'])
		print(df_plag)

		# Plot with seaborn

		fig = plt.figure(figsize=(20,5))

		sns.plt.title(r'Intrinsic plagiarism, ${}-profiling$'.format(analyzer))
		sns.set(font_scale=0.5)
		sns.set_style("darkgrid")

		ax = sns.pointplot(x=df['range'], y=df['dissimilarity measure'], data=df, ci=5, scale=0.4)
		ax.set_xlabel(r"Step Size: ${}$ {}s, Sample Size: ${}$ {}s".format(str(self.step_size), analyzer, str(self.sample_size), analyzer))
		ax.set_ylabel(r"Dissimilarity measure ($d$)")

		# Plot red line
		plt.plot([0, step_ranges[-1][-1]], [filter_threshold, filter_threshold], '--', lw=0.75, color='r')

		# Set right xtick labels
		increment = np.int(np.round(len(ax.get_xticklabels())/10))
		visible_labels = range(0, len(ax.get_xticklabels()), increment)
		for idx, label in enumerate(ax.get_xticklabels()):
		    if idx in visible_labels:
		    	label.set_visible(True)
		    else:
		    	label.set_visible(False)

		sns.plt.show()
		fig.savefig("/Users/jedgusse/compstyl/output/fig_output/intrinsic_plagiarism.pdf", bbox_inches='tight')

class HeatMap:

	def __init__(self, doc_vectors, features, authors, titles):
		self.doc_vectors = doc_vectors
		self.authors = authors
		self.titles = titles
		self.features = features 

	def plot(self):
		distance_matrix = squareform(pdist(self.doc_vectors, 'cityblock'))
		