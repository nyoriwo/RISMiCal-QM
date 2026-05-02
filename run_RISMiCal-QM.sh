#!/bin/bash
### 
#PBS -q cpu
#PBS -l select=1:ncpus=20
#PBS -j oe

cd $PBS_O_WORKDIR

#
# usage: qsub -v1 arg1=inputfile.inp run_RISMiCal-QM.sh 
#

output=`basename $arg1 .inp`.out

python3 ~/software/RISMiCal-QM/RISMiCal-QM.py $arg1 &> $output
