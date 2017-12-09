CXX=nvcc
CXXFLAGS=-std=c++14 -O3

all: a3

demo:
	nvcc -std=c++14 -O3 a3.cu -o a3
	nvprof ./a3 262144 1

clean:
	rm -rf a3
