CC = g++
CCFLAGS = -fPIC -O3 -Wall -pedantic -ffast-math -msse -msse2 -fopenmp
LINKFLAGS = -shared -s -Wl -lgomp
INPUT = im2col_splitchannel.cpp im2col_mergedchannel.cpp
TARGET = libcpputil.so

all:
	$(CC) -c $(CCFLAGS) $(INPUT)
	$(CC) $(LINKFLAGS) -o $(TARGET) *.o
clean:
	rm *.so
	rm *.o
