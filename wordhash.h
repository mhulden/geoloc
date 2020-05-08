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
/**************************************************************************/

/* Hash functions for words                          */
/* Stores word in hash and generates a running value */
/* Uses a linear probing table and rehashes          */
/* when table occupancy reaches 0.5                  */
/* MH 20140205                                       */

struct wordhash {
    struct wordhash_table *table;
    unsigned int tablesize;
    unsigned int occupancy;
};

struct wordhash_table {
    char *word;
    int value;
};

struct wordhash *wordhash_init(int tablesize);        /* Initialize hash with tablesize entries             */
int wordhash_insert(struct wordhash *wh, char *word); /* Insert word into hash, value automatically generated */
int wordhash_find(struct wordhash *wh, char *word);   /* Find value for word, -1 = not found                  */
void wordhash_free(struct wordhash *wh);              /* Release memory                                     */
void wordhash_set_value(struct wordhash *wh, char *word, int value);
void wordhash_inc_value(struct wordhash *wh, char *word);

unsigned int wordhash_hashf(char *word) { /* djb2 */
    unsigned int hash = 5381;
    int c;
    while ((c = *word++))
	hash = ((hash << 5) + hash) + c;
    return hash;
}

int wordhash_find(struct wordhash *wh, char *word) {
    struct wordhash_table *table;
    unsigned int hash, j;
    table = wh->table;
    hash = wordhash_hashf(word) % wh->tablesize;
    for (j = 0; j < wh->tablesize; j++) {
	if ((table + hash)->value == - 1)
	    return -1;
	if (strcmp((table+hash)->word, word) == 0) {
	    return((table + hash)->value);
	}
	hash++;
	if (hash >= wh->tablesize)
	    hash -= wh->tablesize;
    }
    return -1;
}

void wordhash_insert_with_value(struct wordhash *wh, char *word, int value) {
    struct wordhash_table *table;
    unsigned int hash;
    table = wh->table;
    hash = wordhash_hashf(word) % wh->tablesize;
    for (;;) {
	if ((table + hash)->value == -1) {
	    (table + hash)->value = value;
	    (table + hash)->word = word;
	    break;
	}
	hash++;
	if (hash >= wh->tablesize)
	    hash -= wh->tablesize;
    }
}

void wordhash_rehash(struct wordhash *wh) {
    int i;
    unsigned int newtablesize, oldtablesize;
    struct wordhash_table *oldtable;
    newtablesize = wh->tablesize * 2;
    oldtablesize = wh->tablesize;
    oldtable = wh->table;
    wh->table = malloc(sizeof(struct wordhash_table) * newtablesize);
    wh->tablesize = newtablesize;
    for (i = 0; i < newtablesize; i++) {
	(wh->table+i)->value = -1;
    }
    for (i = 0; i < oldtablesize; i++) {
	if ((oldtable+i)-> value != -1) {
	    wordhash_insert_with_value(wh, (oldtable+i)->word, (oldtable+i)->value);
	}
    }
    free(oldtable);
}

void wordhash_inc_value(struct wordhash *wh, char *word) {
    int currvalue;
    if ((currvalue = wordhash_find(wh, word)) == -1) {
	wordhash_set_value(wh, word, 1);
    } else {
	wordhash_set_value(wh, word, currvalue+1);
    }
}

void wordhash_set_value(struct wordhash *wh, char *word, int value) {
    struct wordhash_table *table;
    unsigned int hash, j;
    table = wh->table;
    hash = wordhash_hashf(word) % wh->tablesize;
    for (j = 0; j < wh->tablesize; j++) {
	if ((table + hash)->value == - 1) {
	    (table + hash)->value = value;
	    (table + hash)->word = strdup(word);
	    wh->occupancy = wh->occupancy + 1;
	    if (wh->occupancy > wh->tablesize / 2) {
		wordhash_rehash(wh);
	    }
	    return;
	}
	if (strcmp((table+hash)->word, word) == 0) {
	    (table + hash)->value = value;
	    return;
	}
	hash++;
	if (hash >= wh->tablesize)
	    hash -= wh->tablesize;
    }
    exit(1);
}



int wordhash_insert(struct wordhash *wh, char *word) {
    struct wordhash_table *table;
    unsigned int hash;
    table = wh->table;
    hash = wordhash_hashf(word) % wh->tablesize;
    for (;;) {
	if ((table + hash)->value == - 1) {
	    (table + hash)->value = wh->occupancy;
	    (table + hash)->word = strdup(word);
	    wh->occupancy = wh->occupancy + 1;
	    if (wh->occupancy > wh->tablesize / 2) {
		wordhash_rehash(wh);
	    }
	    return(wh->occupancy - 1);
	}
	hash++;
	if (hash >= wh->tablesize)
	    hash -= wh->tablesize;
    }
}

struct wordhash *wordhash_init(int tablesize) {
    struct wordhash *wh;
    int i;
    wh = malloc(sizeof(struct wordhash));
    wh->tablesize = tablesize;
    wh->occupancy = 0;
    wh->table = malloc(sizeof(struct wordhash_table) * wh->tablesize);
    for (i = 0; i < wh->tablesize; i++) {
	(wh->table+i)->value = -1;
    }
    return(wh);
}

void wordhash_free(struct wordhash *wh) {
    unsigned int i;
    if (wh != NULL) {
	if (wh->table != NULL) {
	    for (i = 0; i < wh->tablesize; i++) {
		if ((wh->table+i)->value != -1) {
		    free((wh->table+i)->word);
		}
	    }
	    free(wh->table);
	}
	free(wh);
    }
}
