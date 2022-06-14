# !/bin/sh
# PBS -N app
# PBS -l nodes=8:ppn=1  
/usr/local/bin/mpiexec -np 8 -machinefile $PBS_NODEFILE /home/s2012077/tmp_compile/app
