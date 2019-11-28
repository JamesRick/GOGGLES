import os
import time
import shutil
import numpy as np
import pandas as pd
import pickle as pkl
import argparse as ap
import subprocess as sp
import multiprocessing as mp
import matplotlib.pyplot as plt

#!/bin/bash
#
#SBATCH --job-name=goggles
#SBATCH --output=goggles_output.txt
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
def write_slurm_script(layer_idx_lists, model_names, num_prototypes, dev_set_sizes, cache, dataset_names):
    for dataset_name in dataset_names:
        for layer_idx_list, model_name in zip(layer_idx_lists, model_names):
            for dev_set_size in dev_set_sizes:
                for num_prototype in num_prototypes:
                    for i in range(1, 21):
                        job_name = 'goggles' + '_' + str(i)
                        pid_string = str(os.getpid())
                        version = dataset_name + '_v0'
                        # slurm_filename = 'slurmSubmit' + pid_string + '.sh'
                        slurm_filename = 'slurmSubmit_'\
                        + str(dataset_name) + '_'\
                        + str(model_name) + '_'\
                        + str(dev_set_size) + '_'\
                        + str(num_prototype) + '_'\
                        + 'seed_' + str(i)\
                        + '.sh'

                        print("Running: " + slurm_filename)

                        layer_idx_list_str = ''
                        for layer in layer_idx_list:
                            layer_idx_list_str = layer_idx_list_str + '"' + str(layer) + '" '

                        print(layer_idx_list_str[:-1])

                        python_command = 'python test_audio.py'\
                                        + ' --layer_idx_list ' + str(layer_idx_list_str[:-1])\
                                        + ' --num_prototypes ' + str(num_prototype)\
                                        + ' --dev_set_size ' + str(dev_set_size)\
                                        + ' --model_name ' + str(model_name)\
                                        + ' --cache ' + str(cache)\
                                        + ' --dataset_name ' + str(dataset_name)\
                                        + ' --version ' + str(version)\
                                        + ' --seed ' + str(i)
                        ###
                        print("Command: " + python_command)

                        os.makedirs('submit_scripts', exist_ok=True)
                        os.makedirs('submit_scripts/output', exist_ok=True)
                        slurm_submit_filename = os.path.join('submit_scripts', slurm_filename)
                        slurm_output_filename = os.path.join('submit_scripts/output', slurm_filename)
                        job_filename = job_name + '.' + slurm_filename

                        with open(slurm_submit_filename, 'w') as slurmFile:
                            slurmFile.write('#!/bin/bash\n')
                            slurmFile.write('#\n')
                            slurmFile.write('#SBATCH --job-name=' + job_filename + '\n')
                            slurmFile.write('#SBATCH --output=' + slurm_output_filename + '.out\n')
                            slurmFile.write('#SBATCH -N 1\n')
                            slurmFile.write('#SBATCH -c 1\n')
                            # slurmFile.write('#SBATCH --ntasks=1\n')
                            slurmFile.write('#SBATCH --mem-per-cpu=2G\n')
                            slurmFile.write('#\n')
                            slurmFile.write(python_command)

                        time.sleep(2.5)
                        sp.call(['sbatch', slurm_submit_filename])


def main():
    dataset_names = ['ESC-10']
    layer_idx_lists = [[2, 5, 10, 15], [3,7,17]]
    model_names = ['vggish', 'soundnet']
    num_prototypes = np.arange(1, 11)
    # dev_set_sizes = [1, 2, 3, 4, 5, 10, 15, 20]
    dev_set_sizes = [5]
    cache = True

    write_slurm_script(layer_idx_lists, model_names, num_prototypes, dev_set_sizes, cache, dataset_names)
    print("Completed Submissions")
    print("Sleeping for 60 seconds")
    time.sleep(60)


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    # parser.add_argument('--layer_idx_list',
    #                     type=int,
    #                     nargs='+',
    #                     default=[3,7,17],
    #                     required=False)

    args = parser.parse_args()
    # main(**args.__dict__)
    main()
