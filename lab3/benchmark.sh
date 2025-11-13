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
    
    # Run CPU version and measure time (3 runs, take average)
    echo "Running CPU version (3 runs)..."
    CPU_TOTAL=0
    for run in 1 2 3; do
        CPU_START=$(date +%s.%N)
        ./seqprimes $N 2>/dev/null
        CPU_END=$(date +%s.%N)
        CPU_RUN=$(echo "$CPU_END - $CPU_START" | bc)
        CPU_TOTAL=$(echo "$CPU_TOTAL + $CPU_RUN" | bc)
        rm -f ${N}.txt
    done
    CPU_TIME=$(echo "scale=4; $CPU_TOTAL / 3" | bc)
    echo "CPU Average Time: ${CPU_TIME}s"
    
    # Run GPU version and measure time (3 runs, take average)
    echo "Running GPU version (3 runs)..."
    GPU_TOTAL=0
    GPU_SUCCESS=true
    for run in 1 2 3; do
        GPU_START=$(date +%s.%N)
        ./genprimes $N 2>/dev/null
        GPU_EXIT=$?
        GPU_END=$(date +%s.%N)
        
        if [ $GPU_EXIT -ne 0 ]; then
            echo "Warning: GPU run $run failed!"
            GPU_SUCCESS=false
            break
        fi
        
        GPU_RUN=$(echo "$GPU_END - $GPU_START" | bc)
        GPU_TOTAL=$(echo "$GPU_TOTAL + $GPU_RUN" | bc)
        rm -f ${N}.txt
    done
    
    if [ "$GPU_SUCCESS" = true ]; then
        GPU_TIME=$(echo "scale=4; $GPU_TOTAL / 3" | bc)
        echo "GPU Average Time: ${GPU_TIME}s"
        
        # Calculate speedup
        SPEEDUP=$(echo "scale=2; $CPU_TIME / $GPU_TIME" | bc)
        echo "Speedup: ${SPEEDUP}x"
        
        # Check if speedup makes sense
        if (( $(echo "$SPEEDUP < 0.01" | bc -l) )); then
            echo "WARNING: Suspiciously low speedup - check implementation!"
        fi
        
        # Save to CSV
        echo "$N,$CPU_TIME,$GPU_TIME,$SPEEDUP" >> $OUTPUT_FILE
    else
        echo "SKIPPING N=$N due to GPU errors"
        echo "$N,$CPU_TIME,ERROR,N/A" >> $OUTPUT_FILE
    fi
    
    echo ""
done

echo "=========================================="
echo "Benchmark complete!"
echo "Results saved to: $OUTPUT_FILE"
echo "Run 'make plot' to generate speedup graph"
echo "=========================================="

# Check for suspicious results
echo ""
echo "Checking for anomalies..."
awk -F',' 'NR>1 && $4 != "N/A" && $4 < 0.1 {print "WARNING: N=" $1 " has suspiciously low speedup of " $4 "x"}' $OUTPUT_FILE
awk -F',' 'NR>1 && $4 != "N/A" && $4 > 1000 {print "WARNING: N=" $1 " has suspiciously high speedup of " $4 "x"}' $OUTPUT_FILE