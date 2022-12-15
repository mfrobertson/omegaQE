#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=30
#SBATCH --nodes=1
#SBATCH --constraint=haswell

module load python
conda activate omegaqe
cd $SCRATCH/repos/omegaQE/kkomega/

TYP="k"
EXP="SO"
FIELDS="TEB"
GMV="True"
Lmax=5000
NL2=1000
Ntheta=1000
NLS=32
OUTDIR="$SCRATCH/bias/results/$EXP"
ID="TEST"
srun -N 1 -n 32 -c 2 python _F_L_mpi.py "$TYP" "$EXP" "$FIELDS" "$GMV" "$Lmax" "$NL2" "$Ntheta" "$NLS" "$OUTDIR" "$ID"