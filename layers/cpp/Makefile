CC = g++
CCFLAGS = -fPIC -O3 -Wall -ffast-math -pedantic -msse -msse2 -fopenmp
LINKFLAGS = -shared -s -Wl -lgomp
INPUT = im2col.cpp fastpool.cpp local_response_normalization.cpp
TARGET = libcpputil.so

all: $(INPUT)
	$(CC) -c $(CCFLAGS) $(INPUT)
	$(CC) $(LINKFLAGS) -o $(TARGET) *.o
speedtest_lrn: speedtest_lrn.cpp local_response_normalization.cpp
	$(CC) $(CCFLAGS) -lgomp -o speedtest_lrn speedtest_lrn.cpp local_response_normalization.cpp
clean:
	rm *.so
	rm *.o
