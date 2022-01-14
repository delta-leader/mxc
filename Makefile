
CC	= g++ -std=c++11 -O3 -I.
MPI_CC	= mpicxx -O3 -I.
LC	= -lm

all: domain bodies minblas linalg kernel basis umv solver dist main
	$(MPI_CC) main.o domain.o bodies.o minblas.o linalg.o kernel.o basis.o umv.o solver.o dist.o $(LC)

domain: domain.cxx domain.hxx
	$(CC) -c domain.cxx

bodies: bodies.cxx bodies.hxx
	$(CC) -c bodies.cxx

minblas: minblas.c minblas.h
	gcc -O3 -c minblas.c

linalg: linalg.cxx linalg.hxx
	$(CC) -c linalg.cxx

kernel: kernel.cxx kernel.hxx
	$(CC) -c kernel.cxx

basis: basis.cxx basis.hxx
	$(CC) -c basis.cxx

umv: umv.cxx umv.hxx
	$(CC) -c umv.cxx

solver: solver.cxx solver.hxx
	$(CC) -c solver.cxx

dist: dist.cxx dist.hxx
	$(MPI_CC) -c dist.cxx

main: main.cxx
	$(MPI_CC) -c main.cxx

clean:
	rm -f *.o a.out
