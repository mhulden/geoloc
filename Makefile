PREFIX = /usr/local
BINPREFIX = $(PREFIX)/bin/
CC=cc
CFLAGS=-O3 -std=c99 -Wall
LFLAGS=-lz -lm

geoloc: geoloc.c wordhash.h
	$(CC) $(CFLAGS) -o geoloc geoloc.c $(LFLAGS)

clean:
	rm geoloc

install: geoloc
	-@if [ ! -d $(BINPREFIX) ]; then mkdir -p $(BINPREFIX); fi
	cp geoloc $(BINPREFIX)
