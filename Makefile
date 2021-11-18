
CC	= g++ -O3 -I.
MPI_CC	= mpicxx -O3 -I.
LC	= -lm -llapacke -lblas

all: main build_tree linalg kernel sps_basis sps_umv dist
	$(MPI_CC) main.o build_tree.o linalg.o kernel.o sps_basis.o sps_umv.o dist.o $(LC)

build_tree: build_tree.cxx build_tree.hxx
	$(CC) -c build_tree.cxx

linalg: linalg.cxx linalg.hxx
	$(CC) -c linalg.cxx

kernel: kernel.cxx kernel.hxx
	$(CC) -c kernel.cxx

sps_basis: sps_basis.cxx sps_basis.hxx
	$(CC) -c sps_basis.cxx

sps_umv: sps_umv.cxx sps_umv.hxx
	$(CC) -c sps_umv.cxx

dist: dist.cxx
	$(MPI_CC) -c dist.cxx

main: main.cxx
	$(MPI_CC) -c main.cxx

clean:
	rm -f *.o a.out