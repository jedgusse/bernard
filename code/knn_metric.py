#!/usr/bin/env

from sklearn.neighbors import NearestNeighbors
import numpy as np 

def knn(authors, titles, vectors, neighbors):
	
	nbrs = NearestNeighbors(n_neighbors=neighbors, algorithm='ball_tree').fit(vectors)
	distances, indices = nbrs.kneighbors(vectors)
	
	return distances, indices