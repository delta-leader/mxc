
EIGEN_DIR	= /mnt/nfs/packages/x86_64/eigen/eigen-3.4.0

CXX		= mpicxx -std=c++17
CFLAGS	= -O3 -m64 -Wall -Wextra -Wpedantic -fexceptions -fopenmp -I. -I"${EIGEN_DIR}/include/eigen3"
LDFLAGS	= -lpthread -lm -ldl

OBJDIR	= ./obj

HEADER	= build_tree.hpp comm-mpi.hpp h2matrix.hpp kernel.hpp solver.hpp
SRCS	= build_tree.cpp comm-mpi.cpp h2matrix.cpp kernel.cpp solver.cpp main.cpp

OBJS 	= $(addprefix $(OBJDIR)/,$(patsubst %.cpp,%.o,$(SRCS)))

$(OBJDIR)/%.o: %.cpp $(HEADER)
	$(CXX) -c -o $@ $< $(CFLAGS)

main.app: $(OBJS)
	$(CXX) -o $@ $^ $(CFLAGS) $(LDFLAGS)

$(OBJS): | $(OBJDIR)
 
$(OBJDIR):
	mkdir $(OBJDIR)

.PHONY: clean

clean:
	rm -f $(OBJDIR)/*.o *.app
	rm -r $(OBJDIR)
