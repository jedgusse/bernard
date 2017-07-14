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
import os
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

				ax.scatter(x1, x2, 100, edgecolors='none', facecolors='none', cmap='rainbow')
				for index, (p1, p2, a, title) in enumerate(zip(x1, x2, self.authors, self.titles)):
					ax.scatter(p1, p2, marker='o', color=cmap(color_dict[a]), s=20)
					ax.text(p1, p2, title.split('_')[-1], color='black', fontdict={'size': 5})

				# Legend settings (code for making a legend)

				collected_patches = []
				for author in set(self.authors):
					legend_patch = mpatches.Patch(color=cmap(color_dict[author]), label=author.split('-')[0])
					collected_patches.append(legend_patch)
				plt.legend(handles=collected_patches, fontsize=7)

				ax.set_xlabel('Principal Component 1 \n \n Explained Variance: {}% \n Sample Size: {} words/sample \n Number of Features: {} features'.format(str(explained_variance), str(self.sample_size), str(len(self.features))), fontdict={'size': 7})
				ax.set_ylabel('Principal Component 2', fontdict={'size': 7})

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

				fig.savefig(os.path.dirname(os.getcwd()) + "/pca.pdf", transparent=True, format='pdf')

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
				fig.savefig(os.path.dirname(os.getcwd()) + "/pca.pdf", bbox_inches='tight', transparent=True, format='pdf')

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

			sns_plot.savefig(os.path.dirname(os.getcwd()) + "/pca.pdf")
			
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

		fob_nodes = open(os.path.dirname(os.getcwd()) + "/gephi_nodes.txt", "w")
		fob_edges = open(os.path.dirname(os.getcwd()) + "/gephi_edges.txt", "w")

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

			# Token pattern in order for hyphenated title names not to become split up
			pattern = "(?u)\\b[\\w-]+\\b"
			model = CountVectorizer(lowercase=False, token_pattern=pattern)
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
