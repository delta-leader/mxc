
CC	= g++ -std=c++11 -O3 -I.
MPI_CC	= mpicxx -O3 -I.
LC	= -lm

all: build_tree bodies minblas linalg kernel basis umv h2mv solver dist main
	$(MPI_CC) main.o build_tree.o bodies.o minblas.o linalg.o kernel.o basis.o umv.o h2mv.o solver.o dist.o $(LC)

build_tree: build_tree.cxx build_tree.hxx
	$(CC) -c build_tree.cxx

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

h2mv: h2mv.cxx h2mv.hxx
	$(CC) -c h2mv.cxx

solver: solver.cxx solver.hxx
	$(CC) -c solver.cxx

dist: dist.cxx dist.hxx
	$(MPI_CC) -c dist.cxx

main: main.cxx
	$(MPI_CC) -c main.cxx

clean:
	rm -f *.o a.out
