import os
import re
import time
import shutil
import tqdm
import numpy as np
import pandas as pd
import pickle as pkl
import argparse as ap
import subprocess as sp
import matplotlib.pyplot as plt

from goggles.test.run_audio_2 import load_df

def test_slurm_script(layer_idx_lists, model_names, num_prototypes, dev_set_sizes, cache, dataset_names, classes_list):
    j = 0
    k = 0
    n_i = 0
    for dataset_name, classes in zip(dataset_names, classes_list):
        print(dataset_name)
        for cur_class_list in classes:
            for model_layer_list, model_name in zip(layer_idx_lists, model_names):
                for layer_idx_list in model_layer_list:
                    for dev_set_size in dev_set_sizes:
                        for num_prototype in num_prototypes:
                            for i in range(1, 6):
                                k += 1
                                n_i += 1
    print("Total: " + str(k))
    return k

def call_sacct():
    p1 = sp.Popen(['sacct', '-S2019-12-07-12:50:00', '-o', 'JobID,State,Elapsed,JobName%110'], stdout=sp.PIPE)
    p2 = sp.Popen(['grep', '-E', "RUNNING|PENDING"], stdin=p1.stdout, stdout=sp.PIPE)
    p1.stdout.close()
    p3 = sp.Popen(['wc', '-l'], stdin=p2.stdout, stdout=sp.PIPE)
    p2.stdout.close()
    print("Running communicate")
    num_cur_jobs, err = p3.communicate()
    print("Communicate Complete")
    p3.stdout.close()
    num_cur_jobs = re.sub('\\n', '', num_cur_jobs.decode('UTF-8'))
    print(num_cur_jobs)
    p1.wait(); p2.wait(); p3.wait()
    p1.kill(); p2.kill(); p3.kill()
    print("Current Number of Jobs Submitted: ", num_cur_jobs)
    return num_cur_jobs

def call_squeue():
    print("Running wait to submit")
    p1 = sp.Popen(['squeue', '-u', 'jrick6', '-t', 'PENDING,RUNNING'], stdout=sp.PIPE)
    p2 = sp.Popen(['wc', '-l'], stdin=p1.stdout, stdout=sp.PIPE)
    p1.stdout.close()
    print("Running communicate")
    num_cur_jobs, err = p2.communicate()
    print("Communicate Complete")
    num_cur_jobs = re.sub('\\n', '', num_cur_jobs.decode('UTF-8'))
    print(num_cur_jobs)
    p1.wait(); p2.wait()
    p1.kill(); p2.kill()
    return num_cur_jobs

def wait_to_submit():
    print("Running wait to submit")
    try:
        num_cur_jobs = call_squeue()
    except Exception as e:
        print(str(e))
        time.sleep(5)
        num_cur_jobs = call_sacct()
    print("Current Number of Jobs Submitted: ", num_cur_jobs)
    while int(num_cur_jobs) > 300:
        for i in tqdm.trange(10, leave=False):
            time.sleep(3)
        try:
            num_cur_jobs = call_squeue()
        except Exception as e:
            print(str(e))
            time.sleep(1)
            num_cur_jobs = call_sacct()
        print("Current Number of Jobs Submitted: ", num_cur_jobs)

def write_slurm_script(layer_idx_lists, model_names, num_prototypes, dev_set_sizes, cache, dataset_names, classes_list, total):
    j = 0
    for dataset_name, classes in zip(dataset_names, classes_list):
        for cur_class_list in classes:
            for model_layer_list, model_name in zip(layer_idx_lists, model_names):
                for layer_idx_list in model_layer_list:
                    wait_to_submit()
                    for dev_set_size in dev_set_sizes:
                        time.sleep(0.1)
                        for num_prototype in num_prototypes:
                            for i in range(1, 6):
                                job_name = 'goggles' + '_' + str(i)
                                pid_string = str(os.getpid())
                                version = 'v7'
                                layer_idx_list_str = ''
                                for layer in layer_idx_list:
                                    layer_idx_list_str = layer_idx_list_str + '"' + str(layer) + '" '

                                classes_str = ''
                                for class_name in cur_class_list:
                                    classes_str = classes_str + '"' + str(class_name) + '" '

                                slurm_layer_filename = re.sub(" ", "-", re.sub('"', "", layer_idx_list_str[:-1]))
                                slurm_classes_filename = re.sub(" ", "-", re.sub('"', "", classes_str[:-1]))
                                slurm_filename = 'slurmSubmit_'\
                                + str(dataset_name) + '_'\
                                + str(model_name) + '_'\
                                + str(slurm_layer_filename) + '_'\
                                + str(slurm_classes_filename) + '_'\
                                + str(dev_set_size) + '_'\
                                + str(num_prototype) + '_'\
                                + 'seed_' + str(i)\
                                + '.sh'

                                print("Running: " + slurm_filename)

                                python_command = 'python run_audio.py'\
                                                + ' --layer_idx_list ' + str(layer_idx_list_str[:-1])\
                                                + ' --num_prototypes ' + str(num_prototype)\
                                                + ' --dev_set_size ' + str(dev_set_size)\
                                                + ' --model_name ' + str(model_name)\
                                                + ' --cache ' + str(cache)\
                                                + ' --dataset_name ' + str(dataset_name)\
                                                + ' --version ' + str(version)\
                                                + ' --seed ' + str(i)\
                                                + ' --classes ' + classes_str[:-1]
                                ###
                                print("Command: " + python_command)
                                dataset_output = os.path.join('submit_logs', dataset_name + '_' + model_name + '_output')
                                submit_script = os.path.join('submit_scripts', dataset_name + '_' + model_name + '_script')
                                os.makedirs(dataset_output, exist_ok=True)
                                os.makedirs(submit_script, exist_ok=True)
                                slurm_submit_filename = os.path.join(submit_script, slurm_filename)
                                slurm_output_filename = os.path.join(dataset_output, slurm_filename)
                                job_filename = job_name + '.' + slurm_filename

                                with open(slurm_submit_filename, 'w') as slurmFile:
                                    slurmFile.write('#!/bin/bash\n')
                                    slurmFile.write('#\n')
                                    slurmFile.write('#SBATCH --job-name=' + job_filename + '\n')
                                    slurmFile.write('#SBATCH --output=' + slurm_output_filename + '.out\n')
                                    slurmFile.write('#SBATCH -c 1\n')
                                    slurmFile.write('#SBATCH -t 7-12:00\n')
                                    slurmFile.write('#SBATCH --exclude=ice[107-122,143-145,149,151-153,160,161,162-165]\n')
                                    # slurmFile.write('#SBATCH --gres=gpu')
                                    slurmFile.write('#SBATCH --mem-per-cpu=4G\n')
                                    slurmFile.write('#\n')
                                    slurmFile.write(python_command)

                                # time.sleep(0.1)
                                sp.call(['sbatch', slurm_submit_filename])
                                j += 1
                                print(j, "/", total)


def main(num_classes=2):
    np.random.seed(715)
    # num_classes = 2
    goggles_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # dataset_names = ['ESC-10', 'ESC-50', 'UrbanSound8K', 'TUT-UrbanAcousticScenes', 'LITIS']
    # dataset_names = ['ESC-10', 'ESC-50'] # , 'UrbanSound8K', 'TUT-UrbanAcousticScenes']
    # dataset_names = ['ESC-10', 'ESC-50', 'UrbanSound8K']
    # dataset_names = ['ESC-50', 'ESC-10']
    # dataset_names = ['ESC-10']
    # dataset_names = ['UrbanSound8K', 'TUT-UrbanAcousticScenes', 'ESC-10', 'ESC-50', 'LITIS']
    # dataset_names = ['ESC-50', 'TUT-UrbanAcousticScenes', 'LITIS', 'ESC-10']
    dataset_names = ['TUT-UrbanAcousticScenes', 'LITIS', 'ESC-10']
    # dataset_names = ['ESC-10', 'ESC-50']
    # dataset_names = ['LITIS']
    classes_list = [[] for x in dataset_names]
    for i, dataset_name in enumerate(dataset_names):
        _, df = load_df(goggles_dir, dataset_name)
        for _ in range(10):
            classes = np.random.choice(np.unique(df['category'].values), size=num_classes, replace=False)
            classes_list[i].append(classes.tolist())
    layer_idx_lists = [[[2, 5, 10, 15]],
                      [[3, 7, 17]],
                      [[17]],
                      [[21]]]
    # layer_idx_lists = [[[2], [5], [10], [15]],
    #                   [[3], [7], [17]]]
    # layer_idx_lists = [[[2,5,10,15]], [[3,7,17]]]
    # layer_idx_lists = [[[3], [7], [17]], [[17]]]
    # model_names = ['vggish', 'soundnet']
    model_names = ['vggish', 'soundnet', 'soundnet_svm', 'vggish_svm']
    num_prototypes = np.arange(1, 17)
    # num_prototypes = np.arange(5, 6)
    dev_set_sizes = np.arange(1, 21)
    # dev_set_sizes = [5]
    cache = True
    import pdb; pdb.set_trace()
    total = test_slurm_script(layer_idx_lists, model_names, num_prototypes, dev_set_sizes, cache, dataset_names, classes_list)
    write_slurm_script(layer_idx_lists, model_names, num_prototypes, dev_set_sizes, cache, dataset_names, classes_list, total)
    print("Completed Submissions")
    print("Sleeping for 5 seconds")
    time.sleep(5)


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('--num_classes',
                        type=int,
                        default=2,
                        required=False)
    args = parser.parse_args()
    main(**args.__dict__)
    main()
