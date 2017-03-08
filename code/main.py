#!/usr/bin/env

from preprocess import vectorize, zscore_Delta, Tfidf_Vectorizer
from knn_metric import knn
from pca import Principal_Components_Analysis
from hierarchy import Dendrogram, Heatmap, Plot_Frequencies, Plot_Loadings, Gephi_Networks
from vocabulary_richness import Deviant_Cwords
from rollingdelta import Rolling_Delta

# Sample length, amount of features, location of corpus folder

sample_length = 1500
feat_amount = 150
nearest_neighbors = 6
folder_location = "/Users/.../"
step_size = 300

invalid_words = ["dummyword", "tu", "tuus", "vester", "vos"]

if __name__ == "__main__":

	authors, titles, texts, raw_counts, features = vectorize(folder_location, sample_length, feat_amount, invalid_words)
	zscore_vectors = zscore_Delta(raw_counts)
	tfidf_vectors = Tfidf_Vectorizer(raw_counts, features)
	distances, indices = knn(authors, titles, zscore_vectors, nearest_neighbors)

	# Terminal scores; report of results
	
	print("\n", "----| Features applied:", "\n")
	print("\t" + ", ".join(features), "\n", "\n")

	#Principal_Components_Analysis(tfidf_vectors, authors, titles, features, show_samples='yes', show_loadings='yes')
	#Dendrogram(zscore_vectors, authors, titles, features)
	#Plot_Loadings(authors, titles, raw_counts, zscore_vectors, features, feat_amount, "Bernard_vester")
	#Plot_Frequencies(authors, titles, raw_counts, features, feat_amount, "Bernard_vester_freq")
	#Heatmap(zscore_vectors, authors, titles, features)
	#Gephi_Networks(authors, titles, tfidf_vectors, nearest_neighbors)
	#Deviant_Cwords(folder_location)
	#Rolling_Delta(sample_length, feat_amount, invalid_words, step_size)


