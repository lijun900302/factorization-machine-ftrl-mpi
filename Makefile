#!/bin/bash
INCLUDEPATH = -I/usr/local/include/ -I/usr/include -I/opt/OpenBLAS/include
LIBRARYPATH = -L/usr/local/lib -L/opt/OpenBLAS/lib
LIBRARY = -lpthread -lopenblas -lm
#train code
CPP_tag = -std=gnu++11

LIB=/home/services/xiaoshu/lib
INCLUDE=/home/services/xiaoshu/include
#train code
all:fm_ftrl_mpi

fm_ftrl_mpi:main.o
	mpicxx $(CPP_tag) -o fm_ftrl_mpi main.o $(LIBRARYPATH) $(LIBRARY)

main.o: src/main.cpp 
	mpicxx $(CPP_tag) $(INCLUDEPATH) -c src/main.cpp 

clean:
	rm -f *~ fm_ftrl_mpi predict *.o
