#!/bin/bash
# run_tests.sh — Automated experiment runner for lab2 heat distribution

TIME_SLEEP=2
EXEC=heatdist
SRC=heatdist.cu

echo "Compiling $SRC ..."
nvcc -O3 -o $EXEC $SRC
if [ $? -ne 0 ]; then
  echo "Compilation failed!"
  exit 1
fi
echo "Build complete."
echo ""

# Define test parameters
SIZES=(100 500 1000 10000)
ITERS=(50 100)

# Output header
echo "=============================================="
echo " Starting Experiments for Heat Distribution "
echo "=============================================="
echo ""

# Loop through iteration counts
for ITER in "${ITERS[@]}"; do
  echo "===== Iterations = $ITER ====="
  for N in "${SIZES[@]}"; do
    echo ""
    echo "--- N = $N ---"

    echo "Running CPU version..."
    ./$EXEC $N $ITER 0
    sleep $TIME_SLEEP

    echo "Running GPU version..."
    ./$EXEC $N $ITER 1
    sleep $TIME_SLEEP
  done
  echo ""
done

echo "=============================================="
echo " All Experiments Completed "
echo "=============================================="