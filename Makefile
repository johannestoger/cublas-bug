default: opt

opt: cublas-bug.cu
	nvcc -lcublas -O3 cublas-bug.cu -o cublas-bug
