# "Bernard of Clairvaux and Nicholas of Montiéramey: A Stylometric Investigation into the Secretary’s Influence"

This repository comprises the code, corpora and graphs for the forthcoming article by Jeroen De Gussem, "Bernard of Clairvaux and Nicholas of Montiéramey: *Tracing the Secretarial Trail with Computational Stylistics*". 

# Corpus

All texts were retrieved from the Patrologia Latina, yet manual corrections in Bernard's texts were made based on Dom Jean Leclercq's editions (SBO, *Sancti Bernardi Opera*). Copyright prevents us from disclosing the corpus texts online, but the data that was gathered from the texts and vectorized to raw counts is nevertheless made available for experiment replication in .csv files. In the article, these raw counts were TF-IDF-normalized, a procedure which divides the function word frequencies by the amount of text samples that respective function word appears in. As a consequence, less common function words received a higher weight which prevents them from sinking away (and losing statistical significance) in between very common function words. For more information, we redirect you to Tf-idf vectorizer function (found in preprocess.py), and to the article itself.

# Code and Visualizations

The code is written in Python, and allows to replicate the experiments and graphs produced in the article. This code is open-source. Feel free to borrow, readjust, correct, etc. We used two statistical techniques to structure and visualize this data. The first is **principal component analysis** (hereafter PCA), the second is ***k* Nearest Neighbours** (hereafter *k*-NN). Their respective results will prove to be similar in a general sense, yet crucially different in the details. We argue that such a double-check in visualization provides for a more accurate, nuanced interpretation and a better intuition of the data. The first is ***k* Nearest Neighbors** (hereafter *k*-NN), the second is **principal component analysis** (hereafter PCA). 

In the first visualization, the ***k*-NN networks**, we first calculated the 5 closest text samples to each text sample by applying k-NN on the frequency vectors. The similarity metric applied for the pairwise distances is the Minkowski metric, a Eucledian metric which is “a very general metric that can be used in a k-NN classifier for any data that is represented as a feature vector,” see Pádraig Cunningham and Sarah Jane Delany, “k-Nearest Neighbour Classifiers,” Multiple Classifier Systems (2007): 4. Accordingly for each text the 5 most similar or closest texts were calculated, weighted in rank of smallest pairwise distance and consequently mapped in space through force-directed graph drawing. The weights were directly derived from the calculated distances. The intuition is then that the distances should be normalized to a (1,0) range. Note that this is not a (0,1) range, since smaller distances correspond to greater similarities and therefore require greater weighting: *distances=(distances – argmin(distances))/(argmin(distances)  - argmax(distances))*. What a k-NN network ultimately captures is which texts are most akin – or have the closest connection – when it comes to writing style. It should be noted that k-NN nearly always finds relationships, as it is much more a closed game. It is designed to link candidates to one another in terms of distance (every text sample needs to find its 5 neighbors), and can presuppose ties which are rather coincidental or non-existent (e.g. in the case of outliers). The network visualization can therefore be biased by a misleading directionality. The algorithm used for the graphs in this article was Force Atlas 2, embedded in GEPHI, an open-source tool for network manipulation and visualization, see Mathieu Bastian et al., “Gephi: An Open Source Software for Exploring and Manipulating Networks,” *Proceedings of the Third International ICWSM Conference* (2009): 361-2. 

Secondly, **PCA** is a technique that allows to reduce a multivariate or multidimensional dataset of many features, such as function word frequencies, to merely 2 or 3 principal components which disregard inconsequential information or noise in the dataset and reveal its important dynamics. The assumption is that the main principal components, our axes in the plot, point in the direction of the most significant change in our data, so that clustering and outliers become clearly visible. Each word in our feature vector is assigned a weighting or loading, which reflects whether or not a word correlates highly with a PC and therefore gains importance as a discriminator in writing style. In a plot, the loadings or function words which overlap with the clustered texts of a particular author are the preferred function words of that author (see fig. 5-6).  PCA is built to find the most meaningful variance of observations along the axes of its principal components. In this sense it is not always interested in finding links between candidates, as k-NN is, but rather in finding links between variables. Disadvantages are that PCA can never explain all the variance of  the data, since it purposefully disregards many features and dimensions which it finds insignificant, and that it has a tendency to produce somewhat nebulous scatterplots when texts are stylistically entangled (as is the case for Bernard and Nicholas). For an elaborate explanation on PCA, see José Nilo G. Binongo and M.W.A. Smith, “The Application of Principal Components Analysis to Stylometry,” *Literary and Linguistic Computing* 14 (1999): 446-66. The PCA plots were generated through the Matplotlib package available for Python, see John D. Hunter, “Matplotlib: A 2D Graphics Environments,” *Computing in Science & Engineering* 9 (2007): 90-95. 

# Acknowledgements

I am much indebted to the wisdom and continuous and patient guidance of prof. dr. Jeroen Deploige, prof. dr. Wim Verbaal and prof. dr. Mike Kestemont, who – each in their respective fields of expertise (cultural medieval history, Latin medieval literature and computational stylistics) – have tremendously inspired and challenged me in writing this piece. Their voices inevitably resound from this text, inasmuch that I cannot solely take credit for the whole. I also warmly thank Dinah Wouters and Micol Long for their thorough reading of my drafts and their helpful suggestions to improve them. 

# Further references

- Mike Kestemont, Jeroen Deploige and Sara Moens, “Collaborative Authorship in the Twelfth Century: A Stylometric Study of Hildegard of Bingen and Guibert of Gembloux,” *Digital Scholarship in the Humanities* (2013): 199-224.
- Mike Kestemont, Jeroen De Gussem, “Integrated Sequence Tagging for Medieval Latin Using Deep Representation Learning,” *Journal of Data Mining and Digital Humanities*, (forthcoming).
