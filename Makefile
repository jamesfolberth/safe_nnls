
SRC := $(shell find . -type f -name "*.cc")
MEX := $(patsubst %.cc, %.mexa64, $(SRC))

CXX := g++
CXXFLAGS += -std=c++11 -Wall -Wextra -pedantic -O2 -march=native -mtune=native -fopenmp
LDLIBS += -lm -ldl -lgomp -lmwblas

all: $(MEX)

%.mexa64: %.cc
	@echo "Building $@ with \`mex\`"
	@mex -largeArrayDims -silent\
		GCC="$(CXX)"\
		CXXFLAGS="\$$CXXFLAGS $(CXXFLAGS)"\
		$< $(LDLIBS)

clean:
	@rm -f $(MEX)

.PHONY: all clean
