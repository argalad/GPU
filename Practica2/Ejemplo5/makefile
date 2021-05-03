CC=gcc
CFLAGS=-I. -O3

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

all: gen_md5 crack_md5

gen_md5: md5.o gen_md5.o
	$(CC) -o gen_md5 md5.o gen_md5.o

crack_md5: md5.o brute_force.o
	$(CC) -o crack_md5 md5.o brute_force.o

clean:
	rm -f gen_md5 crack_md5 *.o


