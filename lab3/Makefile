NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_60
CC = gcc
CFLAGS = -O3 -lm

CUDA_TARGET = genprimes
SEQ_TARGET = seqprimes
CUDA_SOURCE = genprimes.cu
SEQ_SOURCE = seqprimes.c

all: $(CUDA_TARGET) $(SEQ_TARGET)

$(CUDA_TARGET): $(CUDA_SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $(CUDA_TARGET) $(CUDA_SOURCE)

$(SEQ_TARGET): $(SEQ_SOURCE)
	$(CC) $(CFLAGS) -o $(SEQ_TARGET) $(SEQ_SOURCE)

clean:
	rm -f $(CUDA_TARGET) $(SEQ_TARGET) benchmark_results.csv speedup_graph.png
	find . -maxdepth 1 -name '[0-9]*.txt' -delete

test: $(CUDA_TARGET)
	./$(CUDA_TARGET) 10
	./$(CUDA_TARGET) 100
	./$(CUDA_TARGET) 1000
	./$(CUDA_TARGET) 10000

benchmark: all
	@echo "Running comprehensive benchmark..."
	./benchmark.sh

plot: benchmark
	python3 plot_speedup.py

.PHONY: all clean test benchmark plot