# Text-based geolocation

The geoloc tool is a simple command-line supervised geolocation tool. It is written in C.

To compile, run:

```
make
```

To install, run:

```
sudo make install
```

## Basic usage

Normally, you train a classifier using text documents with coordinate information.
use the classifier to geolocate unseen documents.

## Cite

```
@inproceedings{Hulden2015, author = {Hulden, Mans and Silfverberg, Miikka and Francom, Jerid}, booktitle = {Proceedings of the twenty-ninth meeting of the Association for the Advancement of Artificial Intelligence (AAAI)}, title = {{Kernel density estimation for text-based geolocation}}, year = {2015} }
```

## Quick example for the impatient

Train (assumes access to the testcorpus):
```
geoloc --train --longranularity=180 --threshold=5 --nomatrix geoloc-testcorpus/geoloc-testcorpus.gz
```

Classify (assumes access to espmsg.txt):
```
geoloc --classify --longranularity=180 examples/espmsg.txt
```

Plot output:
```
geoloc --classify --print-matrix --longranularity=180 examples/espmsg.txt | python geoplot.py --europe
```

## File formats

In order to train a classifier, you need to provide geoloc with data in an appropriate format. Geoloc requires the training data to be provided in a comma-separated format where each document's features are on a separate line, with the first two fields being the decimal-format latitude and longitude of the document:

```
latitude,longitude,feature_1,feature_2,...,feature_n
```

Here is an example training data file with two documents:

```
42.350771,-83.248981,my,features,are,words,in,this,case
33.624409,-112.239866,but,they,could,be,anything,as,long,as,they,are,comma-separated,strings
```

For this example, we've presumably extracted word features and lowercased them all, but you can you any features at all as long as the feature is representable as a text string. For example, if you wanted to include document publication time as a feature, you could expand the above training data like so:

```
42.350771,-83.248981,PUBTIME_1200,my,features,are,words,in,this,case
33.624409,-112.239866,PUBTIME_1700,but,they,could,be,anything,as,long,as,they,are,comma-separated,strings
```

As another example, we might want to only use letter trigrams as our features, which could be accomplished by perhaps representing the documents like this:

```
42.350771,-83.248981,#my,my#,#fe,fea,eat,atu,ure,res,es#,...
```

Geoloc does none of this preprocessing for you. It is up to the user to provide training and test data in the appropriate format of a text list of features.

Note that you can freely use either gzipped text files or plain text files for the training data.

## Training

The basic usage is

```
geoloc --train training-data.txt
```

Here, it is assumed that training-data.txt contains your training data as per the format above. This will train a model and output it into a default file, which will be `modelXXX.gz`, where `XXX` is the grid granularity (see below). You can set the model file with the option `--modelfile`.

## Grid granularity

The main parameter to set here is the granularity of the grid that the classifier operates on. Geoloc divides the world into discrete cells where the default cell size is 1° x 1°. This corresponds to the option `--longranularity=360`, the circle is divided into 360 'ticks'. To run the above with a 2° x 2° cell size, you would issue:

```
geoloc --train --longranularity=180 training-data.txt
```

And geoloc would output the file model180.gz.

You can use higher granularities, such as 0.1° x 0.1° (which corresponds to `--longranularity=3600`), but be advised that the resulting models can become very large without much gain in accuracy.

## Kernel density estimation

By default, geoloc smoothes out locations over the grid by kernel density estimation. That is, each observation of a feature at some coordinate is assumed to be the mean of a two-dimensional Gaussian. This means that nearby coordinates get some of the observation mass as well. The only parameter to tune here is the width of the Gaussian, which can be set by `--sigma=somevalue` (the default is 3.0).

You can also disable this behavior, using the flag `--nokde`, which may help produce smaller models, at the cost of accuracy.

For example,

```
geoloc --train --longranularity=72 --nokde training-data.txt
```

would train a model which 5° x 5° cells with no kernel density estimation (which should produce a very small model).

## Feature thresholding

You can control the minimum number of times a feature has to be seen to be included in the model. This helps keep the resulting models small, and also helps in terms of accuracy. The default value is 1, but good values are between 3-20, depending on the size of the data. For example:

```
geoloc --train --longranularity=180 --threshold=5 training-data.txt
```

would train a model where all features (words) seen less than 5 times would be discarded.

## Stopwords

You can also include a custom list of stopwords which are ignored using the `--stopwords=FILENAME` flag. The stopwords file should be a list of words, one word per line.

## Space/time tradeoff

You can also issue a `--nomatrix` option, which causes geoloc to not store a density matrix for each word. These will instead be calculated at classification time. This leads to much slower classification, but models remain comparatively small. Probably not worth doing if kernel density estimation is not used (`--nokde`), since those models will be quite small anyway.

## Classification

To classify, use the `--classify` flag. The input data format is assumed to be the same as for training, except the coordinates. That is, just comma-separated lists of features, one line per document. For example:

```
here,is,a,document,i,want,to,classify and,here,is,another,one
```

If you specify a `--longranularity`, geoloc will automatically use the default model filename for that granularity. For example, issuing:

```
geoloc --classify --longranularity=72 unseen-data.txt
```

would use the model file model72.gz (unless you specified one separately with `--modelfile`). Geoloc would then classify all the documents found in unseen-data.txt, and output one coordinate per line:

```
32.5,-87.5
32.5,-117.5
27.5,-97.5
...
```

### Kullback-Leibler (`--kullback-leibler`)

The default classifier is a Naive Bayes classifier. You can also use one based on Kullback-Leibler divergence by issuing the flag `--kullback-leibler`. This is comparable in accuracy to Naive Bayes, but is often slower.

### Centroid classification (`--centroid`)

The default behavior of geoloc is to issue the center coordinate of the most likely cell of the document. It is often better to use, instead of the center, the centroid of the most likely cell. For example, imagine your cell division placed New York City at the corner of a cell, with the center of the cell being in the Atlantic Ocean. Now, by default, each document that was deemed most likely in NYC would actually be issued the cell's center coordinate. To avoid this, we can use the centroid at classification time, which would issue the geometric mean of all the documents seen in the most likely cell.

### Priors (`--prior`)

You can also control the feature prior. This is pseudocount that each cell will have for each feature. For example, when not using kernel density estimation, it is sometimes better to use a higher feature prior than the default 0.01.

### Include unknown features (`--unk`)

If at classification time geoloc runs across an unknown feature never seen at training time, it is simply ignored. You can also try to model unknown features with the `--unk` flag, which would then lump all the unknown features into a single "`UNK`"-feature that would have the same prior as all the other features. Normally, it is better to use the default behavior of skipping unknown features. This is only available for Naive Bayes classification and not for KL-divergence.

### Printing the cell probabilities (`--print-matrix`)
Instead of simply issuing the most likely coordinate for each test document, you can also print out the whole grid, with a probability marked for each cell. For example,

```
geoloc --classify --longranularity=360 --print-matrix unseen-data.txt
```

would print out a matrix of all the cell probabilities, with longranularity/2 lines and longranularity entries per line:

For example, if longranularity = 360, the matrix would look like

```
(x_0,y_0) ... (x_359,y_0) ... (x_0,y_179) ... (x_359,y_179)
```

where `x_0, y_0`, corresponds to the cell with the lower left corner `lon=-180`, `lat=-90` and midpoint `lon=-179.5`, `lat = -89.5`.

The values on each line are tab-separated. The python `geoplot.py`-tool accepts this data format.

## Evaluation

You can also run geoloc in evaluation mode. If you have held-out data that you want to test, geoloc will print out classification accuracy with the `--eval` flag. For example,

```
geoloc --eval --centroid --longranularity=72 heldoutdata.txt
```

would run the classifier on the held-out data, using a previously stored 5° x 5° model (and run the classifier with centroid placement), and print out something like:

```
100: 37.2633,-82.3733 3242.24 1819 running mean: 856.486
200: 40.8773,-73.7268 303.256 1893 running mean: 827.129
...
1800: 40.8773,-73.7268 28.0111 1893 running mean: 794.951

DATA POINTS: 1895 MEAN DISTANCE: 800.263
MEDIAN DISTANCE: 441.089
```

giving the mean and median error (in kilometers) for the data in the heldoutdata.txt.

Note that the held-out data needs to be in the same format as the training data.
