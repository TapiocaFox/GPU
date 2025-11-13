#!/bin/bash

# Benchmark script for CPU vs GPU prime number generation
# Tests N values: 10000, 100000, 1000000, 10000000, 100000000, 1000000000

OUTPUT_FILE="benchmark_results.csv"

# Create CSV header
echo "N,CPU_Time,GPU_Time,Speedup" > $OUTPUT_FILE

# Array of N values to test
N_VALUES=(10000 100000 1000000 10000000 100000000 1000000000)

echo "=========================================="
echo "Prime Number Generation Benchmark"
echo "CPU vs GPU Performance Comparison"
echo "=========================================="
echo ""

for N in "${N_VALUES[@]}"
do
    echo "Testing N = $N"
    echo "------------------------------------------"
    
    # Run CPU version and measure time
    echo "Running CPU version..."
    CPU_START=$(date +%s.%N)
    ./seqprimes $N
    CPU_END=$(date +%s.%N)
    CPU_TIME=$(echo "$CPU_END - $CPU_START" | bc)
    echo "CPU Time: ${CPU_TIME}s"
    
    # Clean up output file
    rm -f ${N}.txt
    
    # Run GPU version and measure time
    echo "Running GPU version..."
    GPU_START=$(date +%s.%N)
    ./genprimes $N
    GPU_END=$(date +%s.%N)
    GPU_TIME=$(echo "$GPU_END - $GPU_START" | bc)
    echo "GPU Time: ${GPU_TIME}s"
    
    # Calculate speedup
    SPEEDUP=$(echo "scale=2; $CPU_TIME / $GPU_TIME" | bc)
    echo "Speedup: ${SPEEDUP}x"
    
    # Save to CSV
    echo "$N,$CPU_TIME,$GPU_TIME,$SPEEDUP" >> $OUTPUT_FILE
    
    # Clean up output file
    rm -f ${N}.txt
    
    echo ""
done

echo "=========================================="
echo "Benchmark complete!"
echo "Results saved to: $OUTPUT_FILE"
echo "Run 'make plot' to generate speedup graph"
echo "=========================================="