/**************************************************************************/
/*   geoloc - feature-based geolocation tool                              */

/*   Geoloc is free software: you can redistribute it and/or modify       */
/*   it under the terms of the GNU General Public License version 2 as    */
/*   published by the Free Software Foundation.                           */

/*   Geoloc is distributed in the hope that it will be useful,            */
/*   but WITHOUT ANY WARRANTY; without even the implied warranty of       */
/*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        */
/*   GNU General Public License for more details.                         */

/*   You should have received a copy of the GNU General Public License    */
/*   along with geoloc.  If not, see <http://www.gnu.org/licenses/>.      */

/*   Author: Mans Hulden <mans.hulden@gmail.com>                          */
/*   MH20140325                                                           */
/**************************************************************************/

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <getopt.h>
#include <float.h>

#include "zlib.h"
#include "wordhash.h"

#define MODE_TRAIN      0
#define MODE_CLASSIFY   1
#define MODE_EVAL       2
#define MODE_TUNE       3

#define MAX_LINE_SIZE 1048576

/* GLOBAL variables */
int g_longranularity = 360;   // Cellgranularity: default is one degree per tick
int g_latgranularity = 180;   // Always g_longranularity/2
int g_use_centroid = 0;       // Whether to use centroids at classification time
int g_unk = 0;                // Whether to model unknown words (1) or to skip them (0)
int g_kullback_leibler = 0;   // Whether to use KL-divergence for classification (default is Naive Bayes)
int g_nokde = 0;              // Skip KDE and just run a "classic" geodesic grid classifier
int g_nomatrix = 0;           // Don't store matrix at all (for smaller model, matrix is computed at class. time)
int g_print_matrix = 0;       // Whether to output the whole matrix at classification time
int g_total_wordcount = 0;    // Total wordcount (tokens)
double g_wordprior = 0.01;    // pseudocounts for words (features)
double g_tweetprior = 1.0;    // pseudocount for tweets (tweets)
int g_threshold = 1;          // Have to see a word/feature this many times to count
double g_sigma = 3.0;         // Defines the covariance of the Gaussian for KDE
unsigned int g_wordtypes = 0; // Number of word types
int g_complement_nb = 0;      // Whether to do complement naive Bayes

static char *versionstring = "Geoloc v1.1";
static char *helpstring =
"\n"
"Train a geolocator and classify text documents on a geodesic grid.\n\n"

" Usage: geoloc [--train|--eval|--classify] [options] DOCUMENTFILENAME\n\n"

"Main options:\n\n"
" -h , --help               Print this help.\n"
" -r , --train              Train a geolocator.\n"
" -C , --classify           Classify documents into cells on the earth.\n"
" -e , --eval               Evaluate performance on dev/test set, with accuracy report.\n"
" -m , --modelfile=FILE     Output model or read model from FILE (otherwise a default name is used).\n\n"

"Training options:\n\n"
" -l , --longranularity=LON Grid size (we divide 360 degrees into LON ticks).\n"
" -n , --nokde              Train a vanilla geodesic grid classifier without kernel density.\n"
" -s , --stopwords=FILE     Read stopwords from FILE (one word per line).\n"
" -S , --sigma=SIGMA        Standard deviation of Gaussians in kernel density estimation.\n"
" -x , --threshold=THR      Must see a word/feature THR times to include in model when training.\n\n"
" -N , --nomatrix           Don't store word matrices = slow classification, but smaller model\n\n"

"Test options:\n\n"
" -k , --kullback-leibler   Use KL-divergence as classification method (instead of Naive Bayes).\n"
" -M , --print-matrix       Print the whole distribution (the grid) at classification time.\n"
" -c , --centroid           Use centroid of most likely cell instead of center.\n"
" -p , --prior              Sets word/feature prior for a cell (default = 0.01).\n"
" -u , --unk                Model unseen words/features instead of just skipping them.\n\n"

"Usage examples:\n\n"
" geoloc --train --longranularity=72  trainingdata.txt\n"
" (train a model with defaults, 5° grid size)\n\n"

" geoloc --train --longranularity=720 --stopwords=stopwords.english --threshold=5  trainingdata.txt\n"
" (train a model with 0.5° grid size, using stopwords, threshold set to 5)\n\n"

" geoloc --train --longranularity=360 --nokde --modelfile=model360nokde  trainingdata.txt\n"
" (train a model with 1° grid size, no kernel density estimation, specify output filename)\n\n"

" geoloc --classify --centroid --prior=0.2 --longranularity=72  testdata.txt\n"
" (classify documents with 0.2 word prior, use default generated filename [model72.gz]\n"
"  and place documents in the most likely cell's centroid [recommended for accuracy])\n\n"

" geoloc --classify --print-matrix --modelfile=mymodel.gz  testdata.txt\n"
" (classify documents in testdata.txt, use model in mymodel.gz, \n print the whole cell distribution for each cell instead of best cell coord)\n\n"

" geoloc --eval --modelfile=model360nokde  testdata.txt\n"
" (evaluate classifier against testdata.txt; prints mean and median error)\n\n"

"File formats:\n\n"
"Training data (--train) is one document per line: LAT,LON,feature1,...,featureN, e.g.:\n"
"42.350771,-83.248981,my,features,are,words,in,this,case\n"
"...\n"
"33.624409,-112.239866,but,they,could,be,anything,as,long,as,they,are,comma-separated,strings\n\n"
"Classification data (--classify) is the same format, except no LAT,LON are given: feature1,...,featureN, e.g.:\n"
    "Test data (--test) is the same as the training data: LAT,LON,feature1,...,featureN, e.g.:\n";

/* We store the centroids for each cell in the matrix */
struct centroids {
    double lat;
    double lon;
};

struct centroids *g_centroids;

/* 

We represent the world as a grid, with the first cell (cell = 0, x = 0, y = 0) 
being -180,-90 (lon,lat), and having it's center at -179.5, -89.5 
(assuming 1 degree granularity where g_longranularity = 360).

     -------------
     |...|...|...|
     -------------
     |360|361|...|
      ------------
     | 0 | 1 |359|
     -------------
     ^ cell# 0 = lon -180.0, lat -90.0 = (x=0,y=0)

*/

/* Macros to convert to and from (lat, lon) <=> (x,y) <=> cell#           */
/* These are needed to convert coordinates into our matrix representation */

// Return cell midpoint in latitude/longitude
#define XTOMIDLON(X) (((double)(X)) * 360.0/(double)g_longranularity - 180.0 + (360.0/(double)g_longranularity)/2.0)
#define YTOMIDLAT(Y) (((double)(Y)) * 360.0/(double)g_longranularity - 90.0 + (360.0/(double)g_longranularity)/2.0)

// Return cell which coordinates fall into
#define LONTOX(LON) ((int)((double)g_longranularity/360.0 * ((LON) + 180.0)))
#define LATTOY(LAT) ((int)((double)g_longranularity/360.0 * ((LAT) + 90.0)))

// Return x or y of cell
#define CELLTOX(CELL) ((CELL) % g_longranularity)
#define CELLTOY(CELL) ((CELL) / g_longranularity)

#define LATLONTOCELL(LAT,LON) (LATTOY(LAT) * g_longranularity + LONTOX(LON))

/* Each tweet's and word/feature's coordinates are stored in a linked list of floats */
struct coordinate_list {
    float lat;
    float lon;
    struct coordinate_list *next;
};

/* Simple sparse matrix format as array , -1 sentinel at x and y */
/* We use float for smaller footprint                            */
/* The corresponding actual matrix is just an array of doubles   */
struct sparsematrix {
    short int x;
    short int y;
    float value;
};

struct sparsematrix_handle {
    struct sparsematrix *sm;
    int size;
    int tail;
};

struct devtraindata {
    char **words;
    double lat;
    double lon;
    struct devtraindata *next;
};

/* The main information we have about a word/feature */
struct wordinfo {
    struct coordinate_list *coordinate_list; // List of coordinates where feature occurs
    struct sparsematrix *sparsematrix;       // a kde matrix with mass in each cell
    char *word;                              // the word/feature itself
    double weight;
    int count;
};

/* Training set word/coordinate list */
struct wordinfo *wc_list; /* Each word index points to a struct with info: (1) coordinate_list (2) kde matrix (3) word */
unsigned int wc_list_size ;
unsigned int wc_list_max = 0;

/* A list of tweet/document's positions of origin is also stored in a plain linked list */
/* This is used to later generate a matrix of densities                                 */
struct coordinate_list *g_tweetcoords_head = NULL;

/* Word hashes for seen words and the stopword list */
struct wordhash *global_wh_train, *global_wh_stopwords = NULL;

double *sparsematrix_to_matrix(struct sparsematrix *sm);
struct sparsematrix *matrix_to_sparsematrix(double *matrix);
double bivariate_gaussian_pdf(double x1, double x2, double sigma1, double sigma2, double rho, double mu1, double mu2);
double quick_pdf (double x, double y, double mu1, double mu2);
void matrix_kde_from_coords(double * restrict matrix, struct coordinate_list *cl, double sigma1, double sigma2, double rho);
double *matrix_init(double prior);
double *matrix_copy(double *matrix1);
void matrix_nokde_from_coords(double * restrict matrix, struct coordinate_list *cl);
void matrix_kde_from_coords(double * restrict matrix, struct coordinate_list *cl, double sigma1, double sigma2, double rho);
void matrix_normalize_log(double *matrix);
double haversine_km(double lat1, double lon1, double lat2, double lon2);
double *word_get_matrix(int wordindex);
void geoloc_write_model(char *modelfilename, double *tweetsmatrix, double *wordmatrix);

/* Add a sparsematrix to a word */
void word_coord_add_sparsematrix(char *word, struct sparsematrix *sm) {
    int wordindex;
    if ((wordindex = wordhash_find(global_wh_train, word)) == -1) {
	fprintf(stderr, "ERROR: word not found when adding sparsematrix!\n");
	exit(EXIT_FAILURE);
    }
    wc_list[wordindex].sparsematrix = sm;
}

void word_coord_set_weight(char *word, double weight) {
    int wordindex;
    if ((wordindex = wordhash_find(global_wh_train, word)) == -1) {
	fprintf(stderr, "ERROR: word not found when setting word weight!\n");
	exit(EXIT_FAILURE);
    }
    wc_list[wordindex].weight = weight;
}

double word_coord_get_count(char *word) {
    int wordindex;
    if ((wordindex = wordhash_find(global_wh_train, word)) == -1) {
	fprintf(stderr, "ERROR: word not found when getting word count!\n");
	exit(EXIT_FAILURE);
    }
    return(wc_list[wordindex].count);
}

double word_coord_get_weight(char *word) {
    int wordindex;
    if ((wordindex = wordhash_find(global_wh_train, word)) == -1) {
	fprintf(stderr, "ERROR: word not found when getting word weight!\n");
	exit(EXIT_FAILURE);
    }
    return(wc_list[wordindex].weight);
}

/* Adds a word/feature and its coordinates to the collection of training data */
void word_coord_add_word(char *word, double lat, double lon, int storeword) {
    struct coordinate_list *cl;
    int wordindex;
    if ((wordindex = wordhash_find(global_wh_train, word)) == -1) {
	wordindex = wordhash_insert(global_wh_train, word);
	wc_list_max = wordindex > wc_list_max ? wordindex : wc_list_max;
	if (wordindex >= wc_list_size) {
	    wc_list = realloc(wc_list, sizeof(struct wordinfo) * wc_list_size * 2);
	    memset(wc_list + wc_list_size, 0, sizeof(struct wordinfo) * wc_list_size);
	    wc_list_size *= 2;
	}
	wc_list[wordindex].sparsematrix = NULL;
	wc_list[wordindex].count = -1;
	if (storeword) { /* Optionally store word, too */
	    wc_list[wordindex].word = strdup(word);
	}
    }
    wc_list[wordindex].count += 1;
    if (lat != 0.0 || lon != 0.0) { /* We can also add entry without coordinates */
	cl = malloc(sizeof(struct coordinate_list));
	cl->next = wc_list[wordindex].coordinate_list;
	wc_list[wordindex].coordinate_list = cl;
	cl->lat = (float)lat;
	cl->lon = (float)lon;
    }
}

int tweet_classify_naivebayes(char **words, double *tweetsmatrix, double *wordmatrix, double *resultmatrix) {
    char **w;
    int maxindex, i, c, wordindex, wordcount;
    double c_iw, p, p_max, c_min, *tempwordmatrix, *totalmatrix, feature_weight;
    /* Shortcut to speed up classification: we don't consider cells that have the minimum prior */
    /* This, unless we want to output the whole distribution  */
    for (c = 0, c_min = 100000; c < g_longranularity * g_latgranularity; c++) {
	if (tweetsmatrix[c] < c_min)
	    c_min = tweetsmatrix[c];
    }

    // Naive Bayes:
    // p(c_i) * mass(c_i, w_1)/mass(c_i)_w * ... * mass(c_i, w_n)/mass(c_i)_w
    /* Create matrices for all words used in tweet */
    totalmatrix = matrix_copy(tweetsmatrix);
    for (c = 0; c < g_longranularity * g_latgranularity ; c++)
	totalmatrix[c] = log(totalmatrix[c]); /* Need to get tweetmatrix in logspace */

    for (w = words, i = 0; *w != NULL; w++) {
	if ((wordindex = wordhash_find(global_wh_train, *w)) != -1) {
	    tempwordmatrix = word_get_matrix(wordindex);
	    feature_weight = word_coord_get_weight(*w);
	    wordcount = word_coord_get_count(*w);
	    i++;
	} else if (g_unk) {
	    /* Unknown word, zero matrix, prior gets added at calc. time below */
	    tempwordmatrix = matrix_init(0.0);
	    i++;
	} else {
	    continue;
	}
	if (feature_weight == 0)
	    continue;

	/* Find argmax c p(c_i) * mass(c_i|w_1)/mass(c_i)_w * ...* mass(c_i|w_n)/mass(c_i)_w */
	for (c = 0; c < g_longranularity * g_latgranularity ; c++) {
	    if (tweetsmatrix[c] == c_min && resultmatrix == NULL)
		continue;
	    if (!g_complement_nb) {
		p = tempwordmatrix[c] + g_wordprior;
		c_iw = log(wordmatrix[c] + g_wordprior * (g_wordtypes + 1.0 + (double)g_unk));  /* p(c)_w + prior (includes UNK) */
		p = log(p) - c_iw;
		totalmatrix[c] += p;
	    } else {
		p = wordcount - tempwordmatrix[c] + g_wordprior; /* Mass in other classes */
		c_iw = log(g_total_wordcount - wordmatrix[c] + g_wordprior * (g_wordtypes + 1.0 + (double)g_unk));  /* p(c)_w + prior (includes UNK) */
		p = log(p) - c_iw;
		totalmatrix[c] -= p;
	    }
	}
	free(tempwordmatrix);
    }
    for (c = 0, p_max = -DBL_MAX, maxindex = 0; c < g_longranularity * g_latgranularity ; c++) {
	if (tweetsmatrix[c] == c_min && resultmatrix == NULL)
	    continue;
	if (totalmatrix[c] > p_max) {
	    p_max = totalmatrix[c];
	    maxindex = c;
	}
    }
    if (resultmatrix != NULL) { /* Copy totalmatrix to resultmatrix */
	for (c = 0; c < g_longranularity * g_latgranularity ; c++)
	    resultmatrix[c] = totalmatrix[c];
    }
    free(totalmatrix);
    return(maxindex);
} 

int tweet_classify_kullbackleibler(char **words, double *tweetsmatrix, double *wordmatrix, double *resultmatrix) {
    char **w, **uniqwords;
    int minindex, i, c, knownwords, wordindex, *seencounts, *wordindices;
    double c_iw, p, p_min, c_min, *totalmatrix, *tempwordmatrix;
    struct wordhash *seenwordhash;
    // KL divergence:
    // sum w \in t p(w|t) * log( p(w|t)/p(w_i|c_i) )
    seenwordhash = wordhash_init(128);
    for (w = words, i = 0; *w != NULL; w++, i++) { } /* Count num features */
    uniqwords = malloc(sizeof(char *) * i);
    wordindices = malloc(sizeof(int) * i);
    for (w = words, i = 0; *w != NULL; w++) {
	if ((wordindex = wordhash_find(global_wh_train, *w)) != -1) {
	    if (wordhash_find(seenwordhash, *w) == -1) {
		uniqwords[i] = *w;
		wordindices[i] = wordindex;
		i++;
	    }
	    wordhash_inc_value(seenwordhash, *w);
	}
    }
    knownwords = i;
    seencounts = malloc(sizeof(int) * knownwords);
    for (i = 0; i < knownwords; i++) {
	seencounts[i] = wordhash_find(seenwordhash, uniqwords[i]);
    }
    /* Shortcut to speed up classification: we don't consider cells that have the minimum prior */    
    /* This, unless we want to output the whole distribution  */
    for (c = 0, c_min = DBL_MAX; c < g_longranularity * g_latgranularity; c++) {
	if (tweetsmatrix[c] < c_min)
	    c_min = tweetsmatrix[c];
    }
    /* p(w|t) ~ 1/knownwords */

    totalmatrix = matrix_init(0.0);
    for (i = 0; i < knownwords; i++) {
	tempwordmatrix = word_get_matrix(wordindices[i]);
	for (c = 0; c < g_longranularity * g_latgranularity ; c++) {
	    if (tweetsmatrix[c] == c_min && resultmatrix == NULL)
		continue;
	    c_iw = wordmatrix[c] + g_wordprior * (g_wordtypes + 1.0 + (double)g_unk);
	    p = seencounts[i] * log((c_iw * seencounts[i])/ (knownwords * (tempwordmatrix[c] + g_wordprior)))/knownwords;
	    totalmatrix[c] += p;
	}
	free(tempwordmatrix);
    }
    for (c = 0, minindex = 0, p_min = DBL_MAX; c < g_longranularity * g_latgranularity ; c++) {
	if (tweetsmatrix[c] == c_min && resultmatrix == NULL)
	    continue;
	if (totalmatrix[c] < p_min) {
	    minindex = c;
	    p_min = totalmatrix[c];
	}
    }
    if (resultmatrix != NULL)
	for (c = 0; c < g_longranularity * g_latgranularity ; c++)
	    resultmatrix[c] = -totalmatrix[c];

    wordhash_free(seenwordhash);
    free(seencounts);
    free(uniqwords);
    free(wordindices);
    return(minindex);
}

int compare_double(const void *a, const void *b) {
       const double *da = (const double *) a;
       const double *db = (const double *) b;     
       return (*da > *db) - (*da < *db);
}

/* Read stopwords and put each word into hash */
void read_stopwords(char *filename) {
    FILE *input_file;
    char line[16384];
    fprintf(stderr, "Reading stopwords from '%s'...\n", filename);
    input_file = fopen(filename, "r");
    if (input_file == NULL) {
        perror(filename);
        exit(EXIT_FAILURE);
    }
    global_wh_stopwords = wordhash_init(128);
    while (fgets(line, 16384, input_file)) {
	strtok(line, "\n");
	if (strlen(line) > 0) {
	    wordhash_insert(global_wh_stopwords, line);
	}
    }
    fclose(input_file);
}

struct wordhash *geoloc_index_words(char *filename) {
    FILE *input_file;
    char *next_field, line[MAX_LINE_SIZE+1];
    struct wordhash *iwh;
    input_file = fopen(filename, "r");
    if (input_file == NULL) {
        perror(filename);
        exit(EXIT_FAILURE);
    }
    iwh = wordhash_init(128);
    for (;;) {
        fgets(line, MAX_LINE_SIZE, input_file);
        if (feof(input_file))
            break;
        next_field = strtok(line, ",\n ");
        while (next_field != NULL) {
	    if (wordhash_find(iwh, next_field) == -1) {
		wordhash_insert(iwh, next_field);
		//fprintf(stderr, "%s\n", next_field);
	    }
	    next_field = strtok(NULL, ",\n ");
        }
    }
    fclose(input_file);
    return(iwh);
}

struct devtraindata *geoloc_read_data(char *filename) {
    FILE *input_file;
    char *next_field, line[MAX_LINE_SIZE+1];
    int linecount, i, j, indmax, line_number, field_number;
    struct devtraindata *data_head, *data;

    input_file = fopen(filename, "r");
    if (input_file == NULL) {
        perror(filename);
        exit(EXIT_FAILURE);
    }
    for (linecount = 0; fgets(line, MAX_LINE_SIZE, input_file); linecount++) {  }
    rewind(input_file);
    data_head = malloc(sizeof(struct devtraindata));
    data_head->next = NULL;
    data = data_head;
    for (line_number = 1, j = 0;  ;line_number++) {
	i = 0;
	if (line_number > 1) {
	    data->next = malloc(sizeof(struct devtraindata));
	    data = data->next;
	    data->next = NULL;
	}
        fgets(line, MAX_LINE_SIZE, input_file);
        if (feof(input_file))
            break;
	field_number = 1;
        next_field = strtok(line, ",\n ");
	data->words = malloc(32 * sizeof(char *));
	indmax = 32;
        while (next_field != NULL) {
	    switch (field_number) {
	    case 1:
		data->lat = strtod(next_field, NULL);
		break;
	    case 2:
		data->lon = strtod(next_field, NULL);
		break;
	    default:
		if (i >= indmax-2) {
		    data->words = realloc(data->words, sizeof(char *) * 2 * indmax);
		    indmax *= 2;
		}
		data->words[i] = strdup(next_field);
		i++;
	    }
	    data->words[i] = NULL;
            next_field = strtok(NULL, ",\n ");
	    field_number++;
        }
    }
    fclose(input_file);
    return(data_head);
}

#define WORDSARRAYSIZE 16
void test_classify(char *filename, double *tweetsmatrix, double *wordmatrix) {
    FILE *input_file;
    char *next_field, line[MAX_LINE_SIZE+1], **words;
    int line_number, field_number, i, x, y, tweet_cell, linecount, wordsarraysize = WORDSARRAYSIZE;
    double lat_estimate, lon_estimate, *resultmatrix;
    words = malloc(sizeof(char *) * wordsarraysize);
    if (g_print_matrix) {
	resultmatrix = calloc(g_longranularity * g_latgranularity, sizeof(double));
    } else {
	resultmatrix = NULL;
    }
    input_file = fopen(filename, "r");
    if (input_file == NULL) {
        perror(filename);
        exit(EXIT_FAILURE);
    }
    for (linecount = 0; fgets(line, MAX_LINE_SIZE, input_file); linecount++) {  }
    rewind(input_file);
    for (line_number = 1;  ;line_number++) {
	i = 0;
        fgets(line, MAX_LINE_SIZE, input_file);
        if (feof(input_file))
            break;
	field_number = 1;
        next_field = strtok(line, ",\n ");
        while (next_field != NULL) {
	    if (i == wordsarraysize) {
		words = realloc(words, sizeof(char *) * wordsarraysize * 2);
		wordsarraysize *= 2;
	    }
	    words[i] = next_field;
	    i++;
            next_field = strtok(NULL, ",\n ");
	    field_number++;
        }
	words[i] = NULL;
	tweet_cell = g_kullback_leibler ? tweet_classify_kullbackleibler(words, tweetsmatrix, wordmatrix, resultmatrix) : tweet_classify_naivebayes(words, tweetsmatrix, wordmatrix, resultmatrix);
	if (g_use_centroid) {
	    lat_estimate = g_centroids[tweet_cell].lat;
	    lon_estimate = g_centroids[tweet_cell].lon;
	} else {
	    lat_estimate = YTOMIDLAT(CELLTOY(tweet_cell));
	    lon_estimate = XTOMIDLON(CELLTOX(tweet_cell));
	}
	if (g_print_matrix) {
	    //fprintf(stderr, "%lg,%lg\n", lat_estimate, lon_estimate);
	    matrix_normalize_log(resultmatrix);
	    for (y = 0; y < g_latgranularity; y++) {
		for (x = 0; x < g_longranularity; x++) {
		    //printf("%lg\t%lg\t%lg\n", YTOMIDLAT(y), XTOMIDLON(x), resultmatrix[x+y*g_longranularity]);
		    printf("%lg", resultmatrix[x+y*g_longranularity]);
		    if (x + 1 < g_longranularity)
			printf("\t");
		}
		printf("\n");
	    }
	    //printf("\n");
	} else {
	    printf("%lg,%lg\n", lat_estimate, lon_estimate);
	}
    }
    free(words);
    fclose(input_file);
}

void geoloc_tune(double *tweetsmatrix, double *wordmatrix, struct devtraindata *dev_data, struct devtraindata *train_data) {
    int i, guess_cell, correct_cell, wordindex;
    double lat_estimate, lon_estimate, error_distance, *tempwordmatrix, correct_cell_weight, guessed_cell_weight, adjust, old_weight, new_weight;
    struct devtraindata *data;

    /* Go through dev data */
    for (data = dev_data; data != NULL; data = data->next) {
	/* Classify tweet with naive bayes */
	guess_cell = tweet_classify_naivebayes(data->words, tweetsmatrix, wordmatrix, NULL);
	correct_cell = LATLONTOCELL(data->lat, data->lon);
	lat_estimate = YTOMIDLAT(CELLTOY(guess_cell));
	lon_estimate = XTOMIDLON(CELLTOX(guess_cell));
	error_distance = haversine_km(data->lat, data->lon, lat_estimate, lon_estimate);
	fprintf(stderr, "GUESSED CELL: %i CORRECT CELL: %i ERROR: %lf\n", guess_cell, correct_cell, error_distance);
	/* Go through each feature if guess is incorrect: */
	if (guess_cell != correct_cell) {
	    for (i = 0; data->words[i] != NULL; i++) {
		/* If word is stored in model, do something, else just skip */
		if ((wordindex = wordhash_find(global_wh_train, data->words[i])) != -1) {
		    tempwordmatrix = word_get_matrix(wordindex);
		    correct_cell_weight = tempwordmatrix[correct_cell];
		    guessed_cell_weight = tempwordmatrix[guess_cell];
		    /* Do something with weight */
		    old_weight = word_coord_get_weight(data->words[i]);
		    adjust = 0.0;
		    if (correct_cell_weight > guessed_cell_weight) {
			adjust = 0.01;
		    } else { 
			adjust = -0.01;
		    }
		    new_weight = old_weight + adjust;
		    word_coord_set_weight(data->words[i], new_weight);
		    free(tempwordmatrix);
		}
	    }
	}
    }
    /* And when we're done: write model to some filename */
    geoloc_write_model("testmodel.gz", tweetsmatrix, wordmatrix);
}


#define WORDSARRAYSIZE 16
void test_evaluate(char *filename, double *tweetsmatrix, double *wordmatrix) {
    FILE *input_file;
    char *next_field, line[MAX_LINE_SIZE+1], **words;
    int line_number, field_number, i, j, tweet_cell, linecount, wordsarraysize = WORDSARRAYSIZE;
    double lat, lon, lat_estimate, lon_estimate, distance, totaldistance = 0.0, *results, median;
    words = malloc(sizeof(char *) * wordsarraysize);
    input_file = fopen(filename, "r");
    if (input_file == NULL) {
        perror(filename);
        exit(EXIT_FAILURE);
    }
    for (linecount = 0; fgets(line, MAX_LINE_SIZE, input_file); linecount++) {  }
    rewind(input_file);
    results = malloc(linecount * sizeof(double));
    for (line_number = 1, j = 0;  ;line_number++) {
	i = 0;
        fgets(line, MAX_LINE_SIZE, input_file);
        if (feof(input_file))
            break;
	field_number = 1;
        next_field = strtok(line, ",\n ");
        while (next_field != NULL) {
	    switch (field_number) {
	    case 1:
		lat = strtod(next_field, NULL);
		break;
	    case 2:
		lon = strtod(next_field, NULL);
		break;
	    default:
		if (i == wordsarraysize) {
		    words = realloc(words, sizeof(char *) * wordsarraysize * 2);
		    wordsarraysize *= 2;
		}
		words[i] = next_field;
		i++;
	    }
            next_field = strtok(NULL, ",\n ");
	    field_number++;
        }
	words[i] = NULL;
	tweet_cell = g_kullback_leibler ? tweet_classify_kullbackleibler(words, tweetsmatrix, wordmatrix, NULL) : tweet_classify_naivebayes(words, tweetsmatrix, wordmatrix, NULL);
	if (g_use_centroid) {
	    lat_estimate = g_centroids[tweet_cell].lat;
	    lon_estimate = g_centroids[tweet_cell].lon;
	} else {
	    lat_estimate = YTOMIDLAT(CELLTOY(tweet_cell));
	    lon_estimate = XTOMIDLON(CELLTOX(tweet_cell));
	}
	distance = haversine_km(lat, lon, lat_estimate, lon_estimate);
	results[j] = distance;
	j++;
	totaldistance += distance;
	if (line_number % 100 == 0)
	    printf("%i: %lg,%lg\t%lg\t%i\trunning mean: %lg\n", line_number, lat_estimate, lon_estimate, distance, tweet_cell, totaldistance/(double)line_number);
    }
    qsort(results, j, sizeof(double), compare_double);
    median = (j % 2 == 0) ? (results[j/2] + results[j/2 - 1])/2 : results[j/2];
    printf("--------------------------\nDATA POINTS: %i\n", line_number - 1);
    printf("MEAN DISTANCE: %lg\n", totaldistance/(double)(line_number-1));
    printf("MEDIAN DISTANCE: %lg\n--------------------------\n", median);
    free(words);
}

/* Read training data and put (1) word information into wc_list and 
                              (2) document origin info into tweetcoords_head */
void training_read(char *filename) {
    gzFile input_file; 
    char *next_field, line[MAX_LINE_SIZE+1], *word;
    int line_number, field_number, i;
    double lat, lon;
    struct coordinate_list *tweetcoords = NULL;
    
    input_file = gzopen(filename, "r");
    if (input_file == NULL) {
        perror(filename);
        exit(EXIT_FAILURE);
    }
    global_wh_train = wordhash_init(128);
    for (line_number = 1, i = 0;  ;line_number++) {
        gzgets(input_file, line, MAX_LINE_SIZE);
        if (gzeof(input_file))
            break;
	field_number = 1;
        next_field = strtok(line, ",\n ");
        while (next_field != NULL) {
	    switch (field_number) {
	    case 1:
		lat = strtod(next_field, NULL);
		break;
	    case 2:
		lon = strtod(next_field, NULL);
		break;
	    default:
		word = next_field;
		if (global_wh_stopwords == NULL || wordhash_find(global_wh_stopwords, word) == -1) {
		    word_coord_add_word(word, lat, lon, 1); /* Add word, lat, lon to table */
		}
	    }
            next_field = strtok(NULL, ",\n ");
	    field_number++;
        }
	/* Store the tweet coordinates */
	if (tweetcoords == NULL) {
	    tweetcoords = malloc(sizeof(struct coordinate_list));
	    tweetcoords->next = NULL;
	    g_tweetcoords_head = tweetcoords;
	} else {
	    tweetcoords->next = malloc(sizeof(struct coordinate_list));
	    tweetcoords = tweetcoords->next;
	    tweetcoords->next = NULL;
	}
	tweetcoords->lat = (float)lat;
	tweetcoords->lon = (float)lon;
    }
    gzclose(input_file);
}

struct sparsematrix_handle *sparsematrix_create() {
    struct sparsematrix_handle *smh;
    smh = malloc(sizeof(struct sparsematrix_handle));
    smh->sm = malloc(128 * sizeof(struct sparsematrix));
    smh->tail = 0;
    smh->size = 128;
    return(smh);
}

void sparsematrix_add(struct sparsematrix_handle *smh, int x, int y, double value) {
    if (smh->tail + 1 == smh->size) {
	smh->sm = realloc(smh->sm, sizeof(struct sparsematrix) * smh->size * 2);
	smh->size *= 2;
    }
    smh->sm[smh->tail].x = (short int)x;
    smh->sm[smh->tail].y = (short int)y;
    smh->sm[smh->tail].value = (float)value;
    smh->tail++;
}

struct sparsematrix *sparsematrix_close(struct sparsematrix_handle *smh) {
    struct sparsematrix *sm;
    smh->sm = realloc(smh->sm, sizeof(struct sparsematrix) * smh->tail);
    sm = smh->sm;
    free(smh);
    return(sm);
}

double *sparsematrix_to_matrix(struct sparsematrix *sm) {
    int i;
    double *matrix;
    matrix = calloc(g_latgranularity * g_longranularity, sizeof(double));
    if (matrix == NULL) {
	fprintf(stderr, "Out of memory.\n");
	exit(EXIT_FAILURE);
    }
    for (i = 0; sm[i].x != -1; i++) {
	matrix[sm[i].x + sm[i].y * g_longranularity] = (double)sm[i].value;
    }
    return(matrix);
}

double *word_get_matrix(int wordindex) {
    double *w;
    /* If matrix is not stored, we generate it on the fly */
    if (wc_list[wordindex].sparsematrix == NULL) {
	w = matrix_init(0.0);
	if (g_nokde) {
	    matrix_nokde_from_coords(w, wc_list[wordindex].coordinate_list);
	} else {
	    matrix_kde_from_coords(w, wc_list[wordindex].coordinate_list, g_sigma, g_sigma, 0.0);
	}
	return(w);
    } else {
	return(sparsematrix_to_matrix(wc_list[wordindex].sparsematrix));
    }
}

struct sparsematrix *matrix_to_sparsematrix(double *matrix) {
    int i, x, y, cellcount;
    struct sparsematrix *sm;
    for (i = 0, cellcount = 0; i < g_latgranularity * g_longranularity; i++)
	if (matrix[i] != 0.0)
	    cellcount++;
    sm = malloc(sizeof(struct sparsematrix) * (cellcount + 1));
    
    for (x = 0, i = 0; x < g_longranularity; x++) {
	for (y = 0; y < g_latgranularity; y++) {
	    if (matrix[x+y*g_longranularity] != 0.0) {
		sm[i].x = (short int)x;
		sm[i].y = (short int)y;
		sm[i].value = (float)matrix[x+y*g_longranularity];
		i++;
	    }
	}
    }
    sm[i].x = -1; sm[i].y = -1; sm[i].value = -1.0;
    return(sm);
}

/* Normalizes a matrix in logspace and converts to reals */
void matrix_normalize_log(double *matrix) {
    int c;
    double max, sum;
    max = -DBL_MAX;
    for (c = 0; c < g_longranularity * g_latgranularity; c++) {
	if (matrix[c] > max)
	    max = matrix[c];
    }
    for (c = 0, sum = 0; c < g_longranularity * g_latgranularity; c++) {
	matrix[c] = exp(matrix[c] - max);
	sum += matrix[c];
    }
    for (c = 0; c < g_longranularity * g_latgranularity; c++) {
	matrix[c] /= sum;
    }
}

void matrix_set(double *matrix, double value) {
    int i;
    for (i = 0; i < g_latgranularity * g_longranularity ; i++) {
	matrix[i] = value;
    }
}

double *matrix_init(double prior) {
    double *matrix;
    matrix = malloc(g_latgranularity * g_longranularity * sizeof(double));
    matrix_set(matrix, prior);
    return(matrix);
}

double *matrix_copy(double *matrix1) {
    int i;
    double *matrix2;
    matrix2 = matrix_init(0.0);
    for (i = 0; i <  g_latgranularity * g_longranularity; i++) {
        matrix2[i] = matrix1[i];
    }
    return(matrix2);
}

void matrix_normalize(double *matrix) {
    int i;
    double sum;
    for (i = 0, sum = 0.0; i < g_latgranularity * g_longranularity; i++) {
	sum += matrix[i];
    }
    for (i = 0; i <  g_latgranularity * g_longranularity; i++) {
	matrix[i] /= sum;
    }
}

void matrix_add(double *matrix1, double *matrix2) {
    int i;
    for (i = 0; i <  g_latgranularity * g_longranularity; i++) {
	matrix2[i] += matrix1[i];
    }
}

/* Find the centroid within each cell for more accurate placement       */
/* at classification time. Takes a list of coordinates, and returns     */
/* a matrix of the centroid for each cell                               */

void find_centroids(struct coordinate_list *cl) {
    struct coordinate_list *clist;
    double *lats, *lons;
    int cell, *counts;
    lats   = calloc(g_latgranularity * g_longranularity, sizeof(double));
    lons   = calloc(g_latgranularity * g_longranularity, sizeof(double));
    counts = calloc(g_latgranularity * g_longranularity, sizeof(int));
    g_centroids = calloc(g_latgranularity * g_longranularity, sizeof(struct centroids));
    for (clist = cl; clist != NULL; clist = clist->next) {
	cell = LONTOX(clist->lon) + LATTOY(clist->lat) * g_longranularity;
	lats[cell] += clist->lat;
	lons[cell] += clist->lon;
	counts[cell] += 1;
    }
    for (cell = 0; cell < g_longranularity * g_latgranularity; cell++) {
	if (counts[cell] == 0) {
	    g_centroids[cell].lat = YTOMIDLAT(CELLTOY(cell));
	    g_centroids[cell].lon = XTOMIDLON(CELLTOX(cell));
	} else {
	    g_centroids[cell].lat = lats[cell]/(double)counts[cell];
	    g_centroids[cell].lon = lons[cell]/(double)counts[cell];
	}
    }
    free(lats);
    free(lons);
    free(counts);
}

/* Non-kde version:                                               */
/* WE just add mass to the matrix for each coordinate in the list */
void matrix_nokde_from_coords(double * restrict matrix, struct coordinate_list *cl) {
    struct coordinate_list * restrict clist;
    int x, y, numpoints;
    
    /* Find out number of points */
    for (clist = cl, numpoints = 0; clist != NULL; clist = clist->next) {
	numpoints++;
    }
    for (clist = cl; clist != NULL; clist = clist->next) {
	x = LONTOX(clist->lon);
	y = LATTOY(clist->lat);
	matrix[x+y*g_longranularity] += 1.0;
    }
}

/* Read list of coordinates (individual points) */
/* and fill values in a density matrix          */
void matrix_kde_from_coords(double * restrict matrix, struct coordinate_list *cl, double sigma1, double sigma2, double rho) {
    double lat, lon, thisdensity;
    struct coordinate_list * restrict clist;
    int x, y, numpoints, maxradius, minx, maxx, miny, maxy;
    /* Find out maximum radius (in ticks) we need to take into account for each point */
    /* This is to avoid having to later calculate the density for the whole grid for each point */
    /* We do this by looking at the Gaussian away from the center until the density dips below a point */ 
    for (x = 0;  ; x++) {
	thisdensity = bivariate_gaussian_pdf((double)x * (360.0/(double)g_longranularity), 0.0, sigma1, sigma2, rho, 0.0, 0.0);
	if (thisdensity < 0.001)
	    break;
    }
    maxradius = x ; /* Maximum radius in ticks worth examining (per degree) */

    /* Find out number of points to examine */
    for (clist = cl, numpoints = 0; clist != NULL; clist = clist->next) {
	numpoints++;
    }
    for (clist = cl; clist != NULL; clist = clist->next) {
	minx = LONTOX(clist->lon) - maxradius;
	minx = minx < 0 ? 0 : minx;
	maxx = LONTOX(clist->lon) + maxradius;
	maxx = maxx >= g_longranularity ? g_longranularity : maxx;
	miny = LATTOY(clist->lat) - maxradius;
	miny = miny < 0 ? 0 : miny;
	maxy = LATTOY(clist->lat) + maxradius;
	maxy = maxy >= g_latgranularity ? g_latgranularity : maxy;
	for (y = miny; y < maxy; y++) {
	    lat = YTOMIDLAT(y);
	    for (x = minx; x < maxx; x++) {
		lon = XTOMIDLON(x);
		/* We measure density at center of cell */
		thisdensity = bivariate_gaussian_pdf(lon, lat, sigma1, sigma2, rho, clist->lon, clist->lat);
		matrix[x+y*g_longranularity] += thisdensity;
	    }
	}
    }
}


/* Get distance in km between two points */

#define TWOPI 6.283185307179586477
#define DEG2RAD(x) (((x) * TWOPI/360))

double haversine_km(double lat1, double lon1, double lat2, double lon2) {
    //double R = 6371; // Radius of the earth in km
    double R = 6372.795;
    double dlat = DEG2RAD(lat2-lat1);
    double dlon = DEG2RAD(lon2-lon1);
    double a = sin(dlat/2) * sin(dlat/2) + cos(DEG2RAD(lat1)) * cos(DEG2RAD(lat2)) * sin(dlon/2) * sin(dlon/2);
    double c = 2 * atan2(sqrt(a), sqrt(1-a));
    return (R * c);
}

/* Density of bivariate gaussian at point x1, x2, given parameters sigma1, sigma2, rho, mu1 (mean), mu2 (mean) */
double bivariate_gaussian_pdf(double x1, double x2, double sigma1, double sigma2, double rho, double mu1, double mu2) {
    double z = (x1 - mu1)*(x1 - mu1)/(sigma1 * sigma1) - 2*rho*(x1 - mu1)*(x2 - mu2)/(sigma1 * sigma2) + (x2 - mu2)*(x2 - mu2)/(sigma2 * sigma2);
    return(1/(TWOPI*sigma1*sigma2*sqrt(1 - rho * rho)) * exp(-(z/(2*(1 - rho * rho)))));
}

int geoloc_read_model(char *modelfilename, double **tm, double **wm, struct wordhash *iwh) {
    double *tweetsmatrix, *wordmatrix, value, lat, lon, feature_weight;
    struct sparsematrix_handle *smh;
    struct sparsematrix *sm;
    int i, x, y, index, has_matrix, numelem; 
    char buf[1024], word[1024];
    gzFile fp;

    /* Reconstruct (1) tweetsmatrix (2) centroid positions (3) wordmatrix (4) wordlist */
    
    fprintf(stderr, "Reading model from %s...\n", modelfilename);
    fp = gzopen(modelfilename, "r");
    gzbuffer(fp, 524288);
    gzgets(fp, buf, 1024);
    if (sscanf(buf, "#LONGRANULARITY# %i", &g_longranularity) != 1) { goto infileerr; }
    g_latgranularity = g_longranularity / 2;
    fprintf(stderr, "Stored model has %i/%i granularity; grid size = %lg° x %lg°\n", g_longranularity, g_latgranularity, (double)360/g_longranularity, (double)360/g_longranularity);

    gzgets(fp, buf, 1024);
    
    /* TWEETMATRIX */
    if (strncmp("#TWEETMATRIX#", buf, 13) != 0)  { goto infileerr; }
    tweetsmatrix = matrix_init(0.0);
    for (;;) {
	gzgets(fp, buf, 1024);
	if (buf[0] == '#')
	    break;
	if (sscanf(buf, "%i %i %lg", &x, &y, &value) != 3) { goto infileerr; }
	tweetsmatrix[x+y*g_longranularity] = value;
    }
    if (strncmp("#END#", buf, 5) != 0)  { goto infileerr; }
    
    gzgets(fp, buf, 1024);
    /* CENTROIDS */
    if (strncmp("#CENTROIDS#", buf, 11) != 0)  { goto infileerr; }
    g_centroids = calloc(g_longranularity * g_latgranularity, sizeof(struct centroids));
    for (i = 0;;i++) {
	gzgets(fp, buf, 1024);
	if (buf[0] == '#')
	    break;
	if (sscanf(buf, "%lg %lg", &lat, &lon) != 2) { goto infileerr; }
	g_centroids[i].lat = lat;
	g_centroids[i].lon = lon;
    }
    if (strncmp("#END#", buf, 5) != 0)  { goto infileerr; }
    
    /* WORDS */
    global_wh_train = wordhash_init(128);
    for (;;) {
	gzgets(fp, buf, 1024);
	if (strncmp("#END#", buf, 5) == 0)
	    break;
	numelem = sscanf(buf, "#WORD# %i %s %lf", &index, word, &feature_weight);
	if (numelem != 2 && numelem != 3) { goto infileerr; }
	if (numelem == 2)
	    feature_weight = 1.0;
	g_wordtypes++;
	if (iwh != NULL && wordhash_find(iwh, word) == -1) {
	    //fprintf(stderr, "SKIP: %s\n", word);
	    for (;;) {
		gzgets(fp, buf, 1024);
		if (strncmp("#END#", buf, 5) == 0)
		    break;
	    }
	} else {
	    //fprintf(stderr, "READ: %s\n", word);
	    word_coord_add_word(word, 0.0, 0.0, 1); /* Add entry, and if coords are provided, add those below */
	    word_coord_set_weight(word, feature_weight);
	    for (;;) {
		gzgets(fp, buf, 1024);
		if (strncmp("#MATRIX#", buf, 8) == 0) {
		    has_matrix = 1;
		    break;
		}
		if (strncmp("#END#", buf, 5) == 0) {
		    has_matrix = 0;
		    break;
		}
		if (sscanf(buf, "%lg %lg", &lat, &lon) != 2) { goto infileerr; }
		/* Add lat lon to word */
		g_total_wordcount++;
		word_coord_add_word(word, lat, lon, 0);
	    }
	    if (has_matrix) {
		smh = sparsematrix_create();
		for (;;) {
		    gzgets(fp, buf, 1024);
		    if (strncmp("#END#", buf, 5) == 0)
			break;
		    if (sscanf(buf, "%i %i %lg", &x, &y, &value) != 3) { goto infileerr; }
		    /* Add x y value to sparsematrix */		
		    sparsematrix_add(smh, x, y, value);
		}
		sparsematrix_add(smh, -1, -1, 0.0);
		sm = sparsematrix_close(smh);
		word_coord_add_sparsematrix(word, sm);
	    } 
	}
    }
    
    /* WORDMATRIX */
    gzgets(fp, buf, 1024);
    if (strncmp("#WORDMATRIX#", buf, 12) != 0) { goto infileerr; }
    wordmatrix = matrix_init(0.0);
    for (;;) {
	gzgets(fp, buf, 1024);
	if (buf[0] == '#')
	    break;
	if (sscanf(buf, "%i %i %lg", &x, &y, &value) != 3) { goto infileerr; }
	wordmatrix[x+y*g_longranularity] = value;
    }
    fprintf(stderr, "Done...\n");
    fprintf(stderr, "Number of word types in model: %i\n", g_wordtypes);
    fprintf(stderr, "Number of word tokens in model: %i\n", g_total_wordcount);
    *tm = tweetsmatrix;
    *wm = wordmatrix;
    return(1);
 infileerr:
    fprintf(stderr, "File error reading model\n");
    return(0);
}

/* Writes a model to a file: assumes tweetsmatrix and wordmatrix are available */
/* Fetches words and coordinates from */
void geoloc_write_model(char *modelfilename, double *tweetsmatrix, double *wordmatrix) {
    gzFile fp;
    int i, j;
    struct sparsematrix *sm;
    struct coordinate_list *cl;

    fp = gzopen(modelfilename, "w");
    fprintf(stderr, "Writing p(c) matrix\n");
    sm = matrix_to_sparsematrix(tweetsmatrix);
    gzprintf(fp, "#LONGRANULARITY# %i\n", g_longranularity);
    gzprintf(fp, "#TWEETMATRIX#\n");
    for (j = 0; sm[j].x != -1; j++)
	gzprintf(fp, "%i %i %lg\n", sm[j].x, sm[j].y, sm[j].value);
    gzprintf(fp, "#END#\n");
    free(sm);
    gzprintf(fp, "#CENTROIDS#\n");
    for (j = 0; j < g_longranularity * g_latgranularity; j++) {
	gzprintf(fp, "%g %g\n", g_centroids[j].lat, g_centroids[j].lon);
    }
    gzprintf(fp, "#END#\n");
    
    for (i = 0; i <= wc_list_max; i++) {
	for (j = 0, cl = wc_list[i].coordinate_list; cl != NULL; cl = cl->next, j++) { }
	if (j < g_threshold)
	    continue;
	/* print #WORD#, followed by word + lats and lons */
	gzprintf(fp, "#WORD# %i %s %lf\n", i, wc_list[i].word, wc_list[i].weight);
	for (cl = wc_list[i].coordinate_list; cl != NULL; cl = cl->next) {
	    gzprintf(fp, "%lg %lg\n", cl->lat, cl->lon);
	}

	sm = wc_list[i].sparsematrix;
	/* print sparse matrix                            */
	if (g_nomatrix == 0) {
	    gzprintf(fp, "#MATRIX#\n");
	    for (j = 0; sm[j].x != -1; j++) {
		gzprintf(fp, "%i %i %lg\n", sm[j].x, sm[j].y, sm[j].value);
	    }
	    free(sm);
	}
	gzprintf(fp, "#END#\n");
    }
    gzprintf(fp, "#END#\n");
    fprintf(stderr, "Writing (unnormalized) p(c)_w matrix...\n");
    sm = matrix_to_sparsematrix(wordmatrix);
    /* Print summed wordmatrix */
    gzprintf(fp, "#WORDMATRIX#\n");
    for (j = 0; sm[j].x != -1; j++) {
	gzprintf(fp, "%i %i %lg\n", sm[j].x, sm[j].y, sm[j].value);
    }
    gzprintf(fp, "#END#\n");
    free(sm);
    gzclose(fp);
}

int geoloc_train_model(char *trainingfilename, char *modelfilename, char *stopwordsfilename, double **tm, double **wm) {
    int i, j;
    gzFile fp;
    double *tweetsmatrix, *wordmatrix, *w;
    struct sparsematrix *sm;
    struct coordinate_list *cl;

    if (stopwordsfilename != NULL)
	read_stopwords(stopwordsfilename);
    
    if (g_nokde) {
	fprintf(stderr, "Not using KDE\n");
    } else {	
	fprintf(stderr, "Using KDE\n");
    }
    fprintf(stderr, "Reading document features/coordinates from training set: '%s'...\n", trainingfilename);
    training_read(trainingfilename);
    
    fp = gzopen(modelfilename, "w");
    if(fp == NULL)
	exit(EXIT_FAILURE);
    
    tweetsmatrix = matrix_init(g_tweetprior); /* Tweet prior must be precalculated into matrix since we normalize */
    fprintf(stderr, "Calculating p(c) matrix...\n");
    if (g_nokde) {
	matrix_nokde_from_coords(tweetsmatrix, g_tweetcoords_head); /* Matrix for p(c) (prior for tweet origin) */
    } else {
	matrix_kde_from_coords(tweetsmatrix, g_tweetcoords_head, g_sigma, g_sigma, 0.0); /* Matrix for p(c) (prior for tweet origin) */
    }
    matrix_normalize(tweetsmatrix);
    fprintf(stderr, "Writing p(c) matrix\n");
    sm = matrix_to_sparsematrix(tweetsmatrix);
    gzprintf(fp, "#LONGRANULARITY# %i\n", g_longranularity);
    gzprintf(fp, "#TWEETMATRIX#\n");
    for (j = 0; sm[j].x != -1; j++)
	gzprintf(fp, "%i %i %lg\n", sm[j].x, sm[j].y, sm[j].value);
    gzprintf(fp, "#END#\n");
    free(sm);
    
    find_centroids(g_tweetcoords_head);
    gzprintf(fp, "#CENTROIDS#\n");
    for (j = 0; j < g_longranularity * g_latgranularity; j++) {
	gzprintf(fp, "%g %g\n", g_centroids[j].lat, g_centroids[j].lon);
    }
    gzprintf(fp, "#END#\n");
    free(g_centroids);
    
    fprintf(stderr, "Number of word types in training set: %i\n", wc_list_max);
    wordmatrix = matrix_init(0.0);
    
    fprintf(stderr, "Calculating p(c)_w matrix...\n");
    w = matrix_init(0.0);
    for (i = 0; i <= wc_list_max; i++) {
	for (j = 0, cl = wc_list[i].coordinate_list; cl != NULL; cl = cl->next, j++) { }
	if (j < g_threshold)
	    continue;
	matrix_set(w, 0.0); /* Word prior is included only at classification time */
	if (i % 5000 == 0)
	    fprintf(stderr, "Calculating p(c|w_i) for i=%i\n", i);

	if (g_nokde) {
	    matrix_nokde_from_coords(w, wc_list[i].coordinate_list);
	} else {
	    matrix_kde_from_coords(w, wc_list[i].coordinate_list, g_sigma, g_sigma, 0.0);
	}
	if (g_nomatrix == 0) {
	    sm = matrix_to_sparsematrix(w);
	}

	/* print #WORD#, followed by word + lats and lons */
	gzprintf(fp, "#WORD# %i %s\n", i, wc_list[i].word);
	for (cl = wc_list[i].coordinate_list; cl != NULL; cl = cl->next) {
	    gzprintf(fp, "%lg %lg\n", cl->lat, cl->lon);
	}
	/* print sparse matrix                            */
	if (g_nomatrix == 0) {
	    gzprintf(fp, "#MATRIX#\n");
	    for (j = 0; sm[j].x != -1; j++) {
		gzprintf(fp, "%i %i %lg\n", sm[j].x, sm[j].y, sm[j].value);
	    }
	    free(sm);
	}
	gzprintf(fp, "#END#\n");
	matrix_add(w, wordmatrix); /* Add this word's mass to total */
    }
    gzprintf(fp, "#END#\n");
    fprintf(stderr, "Writing (unnormalized) p(c)_w matrix...\n");
    sm = matrix_to_sparsematrix(wordmatrix);
    /* Print summed wordmatrix */
    gzprintf(fp, "#WORDMATRIX#\n");
    for (j = 0; sm[j].x != -1; j++) {
	gzprintf(fp, "%i %i %lg\n", sm[j].x, sm[j].y, sm[j].value);
    }
    gzprintf(fp, "#END#\n");
    free(sm);
    gzclose(fp);
    fprintf(stderr, "Wrote model to '%s'.\n", modelfilename);
    *tm = tweetsmatrix;
    *wm = wordmatrix;
    return(1);
}

int main(int argc, char **argv) {
    int opt, option_index = 0, mode = MODE_CLASSIFY, modelspec = 0;
    double *tweetsmatrix, *wordmatrix;
    char *modelfilename, *stopwords = NULL;
    struct wordhash *iwh;
    struct devtraindata *dev_data, *train_data;

    static struct option long_options[] =
	{
	    {"longranularity",    required_argument, 0, 'l'},
	    {"help",                  no_argument  , 0, 'h'},
	    {"train",                 no_argument  , 0, 'r'},
	    {"unk",                   no_argument  , 0, 'u'},
	    {"kullback-leibler",      no_argument  , 0, 'k'},
	    {"stopwords",       required_argument  , 0, 's'},
	    {"sigma",           required_argument  , 0, 'S'},
	    {"eval",                  no_argument  , 0, 'e'},
	    {"nokde",                 no_argument  , 0, 'n'},
	    {"classify",              no_argument  , 0, 'C'},
	    {"centroid",              no_argument  , 0, 'c'},
	    {"print-matrix",          no_argument  , 0, 'M'},
	    {"nomatrix",              no_argument  , 0, 'N'},
	    {"tune",                  no_argument  , 0, 'T'},
	    {"modelfile",       required_argument  , 0, 'm'},
	    {"prior",           required_argument  , 0, 'p'},
	    {"threshold",       required_argument  , 0, 'x'},
	    {0, 0, 0, 0}
	};
    
    while ((opt = getopt_long(argc, argv, "l:hruks:S:enCdcMTNm:p:x:", long_options, &option_index)) != -1) {
	switch(opt) {
	case 'h':
	    printf("%s\n%s", versionstring, helpstring);
	    exit(EXIT_SUCCESS);
	case 'e':
	    mode = MODE_EVAL;
	    break;
	case 'C':
	    mode = MODE_CLASSIFY;
	    break;
	case 'k':
	    g_kullback_leibler = 1;
	    break;
	case 'r':
	    mode = MODE_TRAIN;
	    break;
	case 'T':
	    mode = MODE_TUNE;
	    break;
	case 'm':
	    modelfilename = strdup(optarg);
	    modelspec = 1;
	    break;
	case 'M':
	    g_print_matrix = 1;
	    break;
	case 'N':
	    g_nomatrix = 1;
	    break;
	case 'n':
	    g_nokde = 1;
	    break;
	case 's': 
	    stopwords = strdup(optarg);
	    break;
	case 'c': 
	    g_use_centroid = 1;
	    break;
	case 'u': 
	    g_unk = 1;
	    break;
	case 'l': 
	    g_longranularity = atoi(optarg);
	    g_latgranularity = g_longranularity/2;	   
	    break;
	case 'p':
	    g_wordprior = strtod(optarg, NULL);
	    break;
	case 'S':
	    g_sigma = strtod(optarg, NULL);
	    break;
	case 'x': 
	    g_threshold = atoi(optarg);
	    break;
	}
    }

    argc -= optind;
    argv += optind;

    if (argc == 0) {
	fprintf(stderr, "No document file specified. See geoloc --help\n");
	exit(EXIT_FAILURE);
    }
    if (modelspec == 0) {
	modelfilename = malloc(sizeof(char) * 20);
	snprintf(modelfilename, 20, "%s%i.gz", "model", g_longranularity);
    }
    
    wc_list_size = 1024;
    wc_list = calloc(wc_list_size, sizeof(struct wordinfo));
    
    switch(mode) {
    case MODE_TRAIN:
	fprintf(stderr, "Using %i/%i granularity; grid size = %lg° x %lg°\n", g_longranularity, g_latgranularity, (double)360/g_longranularity, (double)360/g_longranularity);
	geoloc_train_model(argv[0], modelfilename, stopwords, &tweetsmatrix, &wordmatrix);
	break;
    case MODE_EVAL:
	iwh = geoloc_index_words(argv[0]); /* Get an index of words needed from model */
	geoloc_read_model(modelfilename, &tweetsmatrix, &wordmatrix, iwh);
	test_evaluate(argv[0], tweetsmatrix, wordmatrix);
	break;
    case MODE_CLASSIFY:
	iwh = geoloc_index_words(argv[0]); /* Get an index of words needed from model */
	geoloc_read_model(modelfilename, &tweetsmatrix, &wordmatrix, iwh);
	test_classify(argv[0], tweetsmatrix, wordmatrix);
	break;
    case MODE_TUNE:
	geoloc_read_model(modelfilename, &tweetsmatrix, &wordmatrix, NULL);
	dev_data = geoloc_read_data(argv[0]);
	train_data = geoloc_read_data(argv[1]);
	geoloc_tune(tweetsmatrix, wordmatrix, dev_data, train_data);
	break;
    }
}
