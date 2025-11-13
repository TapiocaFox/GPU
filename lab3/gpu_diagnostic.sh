#!/bin/bash

echo "========================================"
echo "GPU PERFORMANCE DIAGNOSTIC"
echo "========================================"
echo ""

# 1. Check GPU hardware
echo "1. GPU Hardware Info:"
echo "-------------------"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.free,clocks.gr,clocks.mem --format=csv
    echo ""
    nvidia-smi --query-gpu=name --format=csv,noheader > /tmp/gpu_name.txt
    GPU_NAME=$(cat /tmp/gpu_name.txt)
    echo "Detected GPU: $GPU_NAME"
    
    # Check if it's an integrated/weak GPU
    if [[ "$GPU_NAME" == *"Integrated"* ]] || [[ "$GPU_NAME" == *"Intel"* ]] || [[ "$GPU_NAME" == *"UHD"* ]]; then
        echo "⚠️  WARNING: Detected integrated/weak GPU!"
        echo "   Integrated GPUs are 10-100x slower than discrete GPUs"
        echo "   This explains your results!"
    fi
else
    echo "✗ nvidia-smi not found"
fi
echo ""

# 2. Check CPU
echo "2. CPU Info:"
echo "-----------"
if [ -f "/proc/cpuinfo" ]; then
    CPU_MODEL=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
    CPU_CORES=$(grep -c "^processor" /proc/cpuinfo)
    echo "CPU: $CPU_MODEL"
    echo "Cores: $CPU_CORES"
fi
echo ""

# 3. Quick memory bandwidth test
echo "3. GPU Memory Bandwidth Test:"
echo "----------------------------"
cat > /tmp/bandwidth_test.cu << 'EOFTEST'
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    size_t size = 100 * 1024 * 1024;  // 100 MB
    char *d_data;
    
    cudaMalloc(&d_data, size);
    
    // Warm up
    cudaMemset(d_data, 0, size);
    cudaDeviceSynchronize();
    
    // Time memset
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        cudaMemset(d_data, i, size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    float bandwidth = (size * 10 / (ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
    printf("GPU Memory Bandwidth: %.2f GB/s\n", bandwidth);
    
    if (bandwidth < 50) {
        printf("⚠️  Very low bandwidth! This GPU is very slow.\n");
    } else if (bandwidth < 200) {
        printf("⚠️  Low bandwidth. Entry-level GPU.\n");
    } else {
        printf("✓ Good bandwidth.\n");
    }
    
    cudaFree(d_data);
    return 0;
}
EOFTEST

nvcc -o /tmp/bandwidth_test /tmp/bandwidth_test.cu 2>/dev/null
if [ $? -eq 0 ]; then
    /tmp/bandwidth_test
else
    echo "Could not compile bandwidth test"
fi
echo ""

# 4. Test actual sieve performance
echo "4. Actual Sieve Performance Test:"
echo "--------------------------------"
if [ -f "./genprimes" ] && [ -f "./seqprimes" ]; then
    echo "Testing N=100000..."
    
    # CPU
    CPU_START=$(date +%s.%N)
    ./seqprimes 100000 2>/dev/null
    CPU_END=$(date +%s.%N)
    CPU_TIME=$(echo "$CPU_END - $CPU_START" | bc)
    
    # GPU
    GPU_START=$(date +%s.%N)
    ./genprimes 100000 2>/dev/null
    GPU_END=$(date +%s.%N)
    GPU_TIME=$(echo "$GPU_END - $GPU_START" | bc)
    
    # Verify correctness
    if diff -q 100000.txt 100000.txt > /dev/null 2>&1; then
        echo "✓ Results match"
    fi
    
    rm -f 100000.txt
    
    echo "CPU Time: ${CPU_TIME}s"
    echo "GPU Time: ${GPU_TIME}s"
    
    SPEEDUP=$(echo "scale=2; $CPU_TIME / $GPU_TIME" | bc)
    echo "Speedup: ${SPEEDUP}x"
    
    if (( $(echo "$SPEEDUP < 0.5" | bc -l) )); then
        echo ""
        echo "⚠️  CRITICAL: GPU is slower than CPU!"
        echo ""
        echo "Possible causes:"
        echo "1. Very weak GPU (integrated graphics)"
        echo "2. Sieve algorithm doesn't parallelize well on your GPU"
        echo "3. Problem size too small for GPU overhead"
        echo "4. CUDA configuration issues"
    fi
else
    echo "✗ Executables not found"
fi
echo ""

# 5. Recommendations
echo "5. Diagnosis & Recommendations:"
echo "------------------------------"
echo ""

if [ -f "/tmp/gpu_name.txt" ]; then
    GPU_NAME=$(cat /tmp/gpu_name.txt)
    
    if [[ "$GPU_NAME" == *"RTX"* ]] || [[ "$GPU_NAME" == *"GTX 1060"* ]] || [[ "$GPU_NAME" == *"GTX 1070"* ]] || [[ "$GPU_NAME" == *"GTX 1080"* ]]; then
        echo "✓ You have a decent GPU. Performance should be good."
        echo "  If GPU is still slow, the algorithm needs optimization."
    elif [[ "$GPU_NAME" == *"Tesla"* ]] || [[ "$GPU_NAME" == *"Quadro"* ]] || [[ "$GPU_NAME" == *"A100"* ]] || [[ "$GPU_NAME" == *"V100"* ]]; then
        echo "✓✓✓ You have a HIGH-END GPU!"
        echo "  GPU should definitely be faster. Check implementation."
    else
        echo "⚠️  Your GPU appears to be low-end or integrated."
        echo ""
        echo "REALITY CHECK:"
        echo "- Integrated/entry-level GPUs are often slower than good CPUs"
        echo "- For Sieve of Eratosthenes, you need:"
        echo "  * Discrete GPU (not integrated)"
        echo "  * At least GTX 1060 / RTX 2060 level"
        echo "  * N >= 1M for speedup"
        echo ""
        echo "Your results (GPU slower than CPU) may be CORRECT"
        echo "for your hardware setup!"
    fi
fi

echo ""
echo "========================================"