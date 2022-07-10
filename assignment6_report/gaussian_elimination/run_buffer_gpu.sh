#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling buffer_gpu
dpcpp buffer_gpu.cpp -o buffer_gpu -O3 -Wall
if [ $? -eq 0 ]; then ./buffer_gpu; fi

