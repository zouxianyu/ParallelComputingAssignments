#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling usm
dpcpp usm.cpp -o usm -O3 -Wall
if [ $? -eq 0 ]; then ./usm; fi

