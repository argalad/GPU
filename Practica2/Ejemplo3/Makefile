
ifndef CC
	CC = gcc
endif

CCFLAGS=-O3 -lm -I/usr/local/cuda-11.3/include

LIBS = -lOpenCL -fopenmp

# Change this variable to specify the device type
# to the OpenCL device type of choice. You can also
# edit the variable in the source.
ifndef DEVICE
	DEVICE = CL_DEVICE_TYPE_DEFAULT
endif

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LIBS = -framework OpenCL
endif

CCFLAGS += -D DEVICE=$(DEVICE)

pi_calc: my_ocl.c pi.c
	$(CC) $^ $(CCFLAGS) $(LIBS) -o $@


clean:
	rm -f pi_calc
