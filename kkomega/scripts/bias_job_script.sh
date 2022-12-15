#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=1200
#SBATCH --nodes=8
#SBATCH --constraint=haswell

module load python
conda activate omegaqe
cd $SCRATCH/omegaQE/kkomega/

NELL=64
FIELDS="TEB"
GMV="True"
BI_TYP="kgI"
EXP="SO"
srun -N 8 -n "$NELL" -c 8 python _bias_mpi.py "$EXP" "$BI_TYP" "$FIELDS" "$GMV" "$NELL" "$SCRATCH/bias/results/$EXP" "kgI8_$EXP"