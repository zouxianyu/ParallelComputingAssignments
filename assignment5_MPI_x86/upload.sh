# !/bin/sh
pssh -h $PBS_NODEFILE mkdir -p /home/s2012077/tmp_compile 
pscp.pssh -h $PBS_NODEFILE /home/s2012077/tmp_compile/app /home/s2012077/tmp_compile
