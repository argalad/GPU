
CXX = g++
CC = gcc
NVCC = nvcc

#OpenCV
CXX_OPENCV_FLAGS+=`pkg-config opencv --cflags`
LD_OPENCV_FLAGS+=`pkg-config opencv --libs`


CFLAGS= -O3 -I.
CXXFLAGS= -O3 -I.

LIBS =-lpng -lm -lcudart

SRC = png_io.o routinesCPU.o routinesGPU.o main.o

all: image

%.o: %.c
	$(CC) $(CFLAGS) -g -c -o $@ $<

%.o: %.cu
	$(NVCC) $(CFLAGS) -G -c -o $@ $<


%.o: %.cpp
	$(CXX) $(CXXFLAGS) -g -c -o $@ $<

image: $(SRC) 
	$(CXX) -o image  $(SRC) $(CXXFLAGS) $(LIBS) 

clean:
	rm -f *.o image
