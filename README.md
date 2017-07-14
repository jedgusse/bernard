# Jeroen De Gussem, "Bernard of Clairvaux and Nicholas of Montiéramey: Tracing the Secretarial Trail with Stylometry", *Speculum* 92.4 (2017): forthcoming October 2017.

This repository comprises the code, corpora and appendix of figures for Jeroen De Gussem, "Bernard of Clairvaux and Nicholas of Montiéramey: Tracing the Secretarial Trail with Stylometry", *Speculum* 92.4 (October 2017), in David Birnbaum, Sheila Bonde and Mike Kestemont, "The *Speculum* Supplement on the Digital Middle Ages", *Speculum* 92.4 (October 2017). 

# Corpus Details

* The analysis relied upon the digitized texts of Bernard of Clairvaux’s *Corpus epistolarum*, *Sermones de diversis*, and *Sermones super Cantica canticorum* as they appear in the state-of-the-art scholarly edition of the *Sancti Bernardi Opera*, ed. Jean Leclercq et al., 8 vols. (Rome, 1957–-77): the *Corpus epistolarum* (*vols.* 7–-8), *Sermones de diversis* (*vol.* 6), and *Sermones super Cantica canticorum* (*vols.* 1-–2). included in the online Brepols Library of Latin Texts. The digitized text files of these editions have been generously provided for our experiments by our project partner, [Brepols Publishers](http://www.brepolis.net).

* For Nicholas of Montiéramey’s letters we are provisionally still reliant on the digitally available *Patrologia Latina* (*PL* 196:1651a–1654b). The sermons have been identified by Jean Leclercq in “Les collections de sermons de Nicolas de Clairvaux,” *Recueil d’études*, 1:52–54. They are collected among those of Peter Damian in *PL* 144, more specifically “Sermo in nativitate S. Ioannis Baptistae” (627), “Sermo in natali apostolorum Petri et Pauli” (649), “Sermo in natali S. Benedicti de evangelio” (548), “Sermo in festivitate S. Mariae Magdalenae” (660), “Sermo in festivitate S. Petri ad vincula” (646), “Sermo in assumptione B. Mariae” (717), “Sermo in nativitate B. Mariae” (736), “Sermo in exaltatione S. crucis” (761), “Sermo in festivitate angelorum” (794), “Sermo in dedicatione ecclesiae” (897), “Sermo in festivate S. Victoris” (732), “Sermo in festivitate omnium sanctorum” (811), “Sermo in festivitate S. Martini” (815), “Sermo in festivitate S. Andreae” (828), “Sermo in festivitate B. Nicholai” (835), “Sermo in festivitate B. Mariae” (557), “Sermo in vigilia nativitatis” (839), “Sermo in nativitate Domini” (847), and “Sermo in festivitate B. Stephani” (853).

All text data are available in the *corpus* folder for experimental replication, yet in a camouflaged form so that the copyright protection on the original text editions is respected. Only the texts’ function words were retained in their original form, whereas all content-loaded words were filtered out and replaced by so-called "dummy words".

Since Leclercq’s editions and the *Patrologia Latina* make use of different orthographical conventions, and since Latin is a synthetic language with a high degree of inflection, Bernard and Nicholas’s lexemes were lemmatized (which means that a specific instance of the word is referred to its headword) and a text’s words (tokens) were classified according to grammatical categories (parts of speech or PoS-tags). For this purpose we applied the [Pandora lemmatizer](https://github.com/mikekestemont/pandora) tagger on the texts, a piece of software developed to achieve specifically this. 

# Code and Visualizations

The code above is written in [Python 3](https://www.python.org/downloads/release/python-360/)), and requires many packages collected under the [Anaconda](https://www.continuum.io/downloads) distribution. 

Clone this repository locally (i.e. download it and place it anywhere). The texts under scrutiny are already in the data folder. All it takes is to download the repository. Run the Python script from the *code* folder in the terminal by typing the following command:

```python main.py```





# Acknowledgements

This article is a result of the research project “Collaborative Authorship in Twelfth-Century Latin Literature: A Stylometric Approach to Gender, Synergy and Authority,” funded by the Ghent University Special Research Fund (BOF). Its execution rests on a close collaboration between the Henri Pirenne Institute for Medieval Studies at Ghent University, the CLiPS Computational Linguistics Group at the University of Antwerp, and the CTLO (Centre Traditio Litterarum Occidentalium) division for computer-assisted research into Latin language and literature housed in the Corpus Christianorum Library and Knowledge Centre of Brepols Publishers in Turnhout (Belgium). I am much indebted to the wisdom and continuous and patient guidance of Jeroen Deploige, Wim Verbaal, and Mike Kestemont, who —each in their respective fields of expertise (medieval cultural history, Latin medieval literature, and computational stylistics)— have tremendously inspired and challenged me in writing this piece. Their voices inevitably resound from this text, so much so that I cannot solely take credit for the whole. I also warmly thank my colleagues from the Latin and History Department in Ghent who have gone through the trouble of reading my preliminary drafts. In particular, Dinah Wouters, Micol Long, and Theo Lap have my sincerest gratitude for personally sending me their valuable feedback. In conclusion, my gratitude goes out to Paul De Jongh, Bart Janssens, Jeroen Lauwers, and Luc Jocqué of Brepols for their commitment to this project.

# References


