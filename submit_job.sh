#!/bin/bash
#PBS -N clique_suite
#PBS -P col7880.ee1221163.course
#PBS -l select=1:ncpus=16:mpiprocs=16
#PBS -l walltime=04:00:00
#PBS -o clique_suite.log
#PBS -e clique_suite.err

module load compiler/gcc/11.2/openmpi/4.1.4

cd ~/distributed-clique-optimization

# Clean up stale CSVs from previous runs so the summary table is clean
rm -f results_*.csv

./test_suite.sh
