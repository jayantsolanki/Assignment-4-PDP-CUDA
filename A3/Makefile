CXX=nvcc
CXXFLAGS=-std=c++14 -O3

all: 
	nvcc -std=c++14 -O3 a3.cu -o a3 #changes cpp to cu extension, saved me frm hassles

demo:
	nvprof ./a3 32 1

clean:
	rm -rf a3
