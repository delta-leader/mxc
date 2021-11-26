
CC	= g++ -O3 -I.
MPI_CC	= mpicxx -O3 -I.
LC	= -lm -llapacke -lblas

all: domain bodies linalg kernel basis umv solver dist main
	$(MPI_CC) main.o domain.o bodies.o linalg.o kernel.o basis.o umv.o solver.o dist.o $(LC)

domain: domain.cxx domain.hxx
	$(CC) -c domain.cxx

bodies: bodies.cxx bodies.hxx
	$(CC) -c bodies.cxx

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
