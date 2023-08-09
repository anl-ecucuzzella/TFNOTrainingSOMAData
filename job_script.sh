#!/bin/bash
#SBATCH  --job-name=compass
#SBATCH  --account=m4259
#SBATCH  --nodes=1
#SBATCH  --output=TR.o%j
#SBATCH  --exclusive
#SBATCH  --time=12:00:00
#SBATCH  --qos=regular
#SBATCH  --constraint=gpu
#SBATCH  --gpus=4

python3 SOMAforward.py

