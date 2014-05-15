all: test

test: test.c
	gcc -o test -g -I../ test.c -L.. -lfault_injector -lpthread

clean:
	rm -f test
