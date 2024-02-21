
CXX		= mpicxx -std=c++17
CFLAGS	= -O3 -m64 -Wall -Wextra -fopenmp -I. -I"${MKLROOT}/include" -D"MKL_ILP64" -D"MKL_Complex16=std::complex<double>"
LDFLAGS	= -L${MKLROOT}/lib -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

OBJDIR	= ./obj

HEADER	= basis.hpp build_tree.hpp comm.hpp geometry.hpp kernel.hpp solver.hpp
SRCS	= main.cpp basis.cpp build_tree.cpp comm.cpp kernel.cpp solver.cpp

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
