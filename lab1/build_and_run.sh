#!/bin/bash
TIME_SLEEP=2

nvcc -o vectorprog vectors.cu -lm
echo "Build complete"

echo "---  Expriment 1 ---"
./vectorprog 1000000 4 500
sleep $TIME_SLEEP
./vectorprog 1000000 8 500
sleep $TIME_SLEEP
./vectorprog 1000000 16 500
sleep $TIME_SLEEP
./vectorprog 1000000 4 250
sleep $TIME_SLEEP
./vectorprog 1000000 8 250
sleep $TIME_SLEEP
./vectorprog 1000000 16 250
sleep $TIME_SLEEP

echo ""
echo "--- Expriment 2 ---"
./vectorprog 100 8 500
sleep $TIME_SLEEP
./vectorprog 1000 8 500
sleep $TIME_SLEEP
./vectorprog 10000 8 500
sleep $TIME_SLEEP
./vectorprog 100000 8 500
sleep $TIME_SLEEP
./vectorprog 1000000 8 500
sleep $TIME_SLEEP
./vectorprog 10000000 8 500
sleep $TIME_SLEEP
./vectorprog 100000000 8 500