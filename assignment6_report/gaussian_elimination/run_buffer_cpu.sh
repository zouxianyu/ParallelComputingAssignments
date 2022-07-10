#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling buffer_cpu
dpcpp buffer_cpu.cpp -o buffer_cpu -O3 -Wall
if [ $? -eq 0 ]; then ./buffer_cpu; fi

