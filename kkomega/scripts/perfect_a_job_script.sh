#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=30
#SBATCH --nodes=1
#SBATCH --constraint=haswell

module load python
conda activate omegaqe
cd $SCRATCH/repos/omegaQE/kkomega/

EXP="SO_base"
NBINS=2
NELL=500
DL2=1
NTHETA=1000
OUTDIR="$SCRATCH/perfect_a/results"
ID="TEST"
srun -N 1 -n 3 -c 8 python perfect_a_mpi.py "$EXP" "$NBINS" "$NELL" "$DL2" "$NTHETA" "$OUTDIR" "$ID"