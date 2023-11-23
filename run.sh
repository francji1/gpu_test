#!/bin/bash

#PBS -N francji1_gpu_test
#PBS -l walltime=05:00:00
#PBS -q gpu
#PBS -j oe
#PBS -l select=1:mem=128G:ncpus=1:ngpus=4

cd /mnt/lustre/helios-home/francji1/WORK/gpu_test/venv/
module load cuda/11.7

# Create and activate virtual environment
if [ ! -d "venv" ]; then
  # Create the virtual environment if it doesn't exist
  python -m venv venv
fi

source /mnt/lustre/helios-home/francji1/WORK/gpu_test/venv/bin/activate
pip install -r requirements.txt

python main.py