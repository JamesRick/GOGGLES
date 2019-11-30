#!/bin/bash
#
#SBATCH --job-name=goggles
#SBATCH --output=goggles_output.txt
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G

python test_audio.py --model_name "vggish"
