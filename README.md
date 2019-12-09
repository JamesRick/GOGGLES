## Installation
This branch uses two submodules for VGGish and SoundNet models. Use the commands below to checkout a copy with both submodules.

```bash
git clone https://github.com/JamesRick/GOGGLES
git checkout audio
git submodule update --init --recursive
cd GOGGLES
pip install -e .
```

SoundNet Submodule forked from: https://github.com/EsamGhaleb/soundNet_pytorch
VGGish Submodule forked from: https://github.com/tcvrick/audioset-vggish-tensorflow-to-pytorch

## Conda Environment
<details>
<summary>Conda Environment Packages</summary>
    \_libgcc_mutex             0.1                        main
    args                      0.1.0                    pypi_0    pypi
    audioread                 2.1.8                    pypi_0    pypi
    bzip2                     1.0.8                h516909a_1    conda-forge
    ca-certificates           2019.9.11            hecc5488_0    conda-forge
    certifi                   2019.9.11                py37_0    conda-forge
    cffi                      1.13.2           py37h8022711_0    conda-forge
    chardet                   3.0.4                    pypi_0    pypi
    clint                     0.5.1                    pypi_0    pypi
    cycler                    0.10.0                   pypi_0    pypi
    decorator                 4.4.1                    pypi_0    pypi
    gettext                   0.19.8.1          hc5be6a0_1002    conda-forge
    goggles                   0.1                       dev_0    <develop>
    idna                      2.8                      pypi_0    pypi
    imageio                   2.6.1                    pypi_0    pypi
    joblib                    0.14.0                   pypi_0    pypi
    kiwisolver                1.1.0                    pypi_0    pypi
    libblas                   3.8.0               14_openblas    conda-forge
    libcblas                  3.8.0               14_openblas    conda-forge
    libffi                    3.2.1             he1b5a44_1006    conda-forge
    libflac                   1.3.1             hf484d3e_1002    conda-forge
    libgcc-ng                 9.1.0                hdf63c60_0
    libgfortran-ng            7.3.0                hdf63c60_2    conda-forge
    liblapack                 3.8.0               14_openblas    conda-forge
    libllvm8                  8.0.1                hc9558a2_0    conda-forge
    libogg                    1.3.2             h14c3975_1001    conda-forge
    libopenblas               0.3.7                h6e990d7_3    conda-forge
    librosa                   0.7.1                    pypi_0    pypi
    libsndfile                1.0.28            hf484d3e_1000    conda-forge
    libstdcxx-ng              9.1.0                hdf63c60_0
    libvorbis                 1.3.5             h14c3975_1001    conda-forge
    llvmlite                  0.30.0           py37h8b12597_1    conda-forge
    matplotlib                3.1.1                    pypi_0    pypi
    ncurses                   6.1               hf484d3e_1002    conda-forge
    networkx                  2.4                      pypi_0    pypi
    numba                     0.46.0           py37hb3f55d8_1    conda-forge
    numpy                     1.17.4                   pypi_0    pypi
    openssl                   1.1.1d               h516909a_0    conda-forge
    pafy                      0.5.4                    pypi_0    pypi
    pandas                    0.25.3                   pypi_0    pypi
    pillow                    6.2.1                    pypi_0    pypi
    pip                       19.3.1                   py37_0    conda-forge
    pycparser                 2.19                     pypi_0    pypi
    pyparsing                 2.4.5                    pypi_0    pypi
    pysoundfile               0.10.2                  py_1001    conda-forge
    python                    3.7.3                h33d41f4_1    conda-forge
    python-dateutil           2.8.1                    pypi_0    pypi
    pytz                      2019.3                   pypi_0    pypi
    pywavelets                1.1.1                    pypi_0    pypi
    readline                  8.0                  hf8c457e_0    conda-forge
    requests                  2.22.0                   pypi_0    pypi
    resampy                   0.2.2                      py_0    conda-forge
    scikit-image              0.16.2                   pypi_0    pypi
    scikit-learn              0.21.3                   pypi_0    pypi
    scipy                     1.3.2            py37h921218d_0    conda-forge
    seaborn                   0.9.0                    pypi_0    pypi
    setuptools                41.6.0                   py37_1    conda-forge
    six                       1.13.0                   py37_0    conda-forge
    sk-video                  1.1.8                    pypi_0    pypi
    sox                       1.3.3                    pypi_0    pypi
    sqlite                    3.30.1               hcee41ef_0    conda-forge
    tk                        8.6.9             hed695b0_1003    conda-forge
    torch                     1.3.0                    pypi_0    pypi
    torchaudio                0.3.1                    pypi_0    pypi
    torchvision               0.4.1                    pypi_0    pypi
    tqdm                      4.38.0                   pypi_0    pypi
    urllib3                   1.25.7                   pypi_0    pypi
    wheel                     0.33.6                   py37_0    conda-forge
    xz                        5.2.4             h14c3975_1001    conda-forge
    youtube-dl                2019.11.5                pypi_0    pypi
    zlib                      1.2.11            h516909a_1006    conda-forge
<br/>

torchaudio is not required. <br/>
python version 3.7 was used. <br/>
</details>


## Data Files
The datasets used can be downloaded from the following links:

ESC-10: https://drive.google.com/file/d/1DyUH55fcSYfzSLjxMAAujXSbT6Gbujdc/view?usp=sharing <br/>
ESC-50: https://drive.google.com/file/d/1mkOOQ0WakozbAKsjhPeMJaTtRng2kgtn/view?usp=sharing <br/>
UrbanSound8K: https://drive.google.com/file/d/1U-lXuzkLSzVN1H_y9bnZdqa0omQa72kW/view?usp=sharing <br/>
TUT-UrbanAcousticScenes: https://drive.google.com/file/d/1qe1gR3B77H_ys52VHXRpJNAD432pxbEe/view?usp=sharing <br/>
LITIS: https://drive.google.com/file/d/19fiBtGrZ_IVWcRdb7-zrrW7-L3Q65TxT/view?usp=sharing <br/>

ESC-10 is the smallest dataset, so this download will be the fastest. <br/>
Download the {dataset-name}.tar.gz file and extract it into the GOGGLES/goggles/data/ folder. <br/>

IMPORTANT NOTE: Datasets are expected to have the directory names as shown above and be located in the GOGGLES/goggles/data folder.

### Examples
tar -xzf {dataset-name}.tar.gz GOGGLES/goggles/data/{dataset-name}

## VGGish weights
The weights for the vggish model are stored in the following link: <br/>
VGGish Weights: https://drive.google.com/file/d/1jsFol7qtN1Nsq3vCPF_rVlyV5OmtugLG/view?usp=sharing <br/>

After completing the download place the pytorch_vggish.pth file in the GOGGLES/goggles/torch_vggish directory. <br/>

IMPORTANT NOTE: These weights are required for running the vggish model and must be placed in the directory as specified above. <br/>

## Results and output files
Results are written to pickle files in the GOGGLES/goggles/output directory. <br/>
These pickle files are parsed with GOGGLES/goggles/utils/postprocessing.py and converted to a csv located in GOGGLES/goggles/results/full_results.csv

Pickle files and full_results csv for the completed experiments can be downloaded here:

Output Directory: https://drive.google.com/file/d/1gvgjfsPBzm-Fq6AJ_hWAVdYWiwzycThr/view?usp=sharing <br/>
Results Directory: https://drive.google.com/drive/folders/12sW_mwsCp11diny2P3z-Q2px97UXIXCt?usp=sharing <br/>

Both of these directories should be placed into the goggles directory.

For quickly running postprocessing.py. Download the results directory and use the following command while inside the utils directory:

    python postprocessing.py --results_csv ../results/full_results.csv

## \_Scratch directory
I've also supplied the cached affinity functions for every run of the experiments in the following link: <br/>
\_scratch: https://drive.google.com/file/d/1DBabZ4KUwdRmyL3dVdB3fNxI2e8kZieD/view?usp=sharing <br/>

Extract this to the GOGGLES/goggles/ directory if you wish to use these cache files.

## Example Usage
Main run script is GOGGLES/goggles/test/run_audio.py <br/>
Default parameters will run SoundNet with ESC-10 dataset. <br/>

Parameters:
--layer_idx_list: The list of max pooling layers to gather affinity functions from. [3, 7, 17] for SoundNet. [2, 5, 10, 15] for VGGish. <br/>
--num_prototypes: Number of prototypes per layer for affinity functions. Default is 10 <br/>
--dev_set_size:   Size of the development set to use for cluster to class inference. Default is 5. <br/>
--model_name:     Name of the model to use. Options: "vggish", "soundnet", "soundnet_svm", "vggish_svm". <br/>
--cache:          Boolean to use cache or not. Default is false, so omit this parameter if you do not wish to use the cache. <br/>
--dataset_name:   Name of the dataset to use. Default is "ESC-10". Options: "ESC-10", "ESC-50", "UrbanSound8K", "TUT-UrbanAcousticScenes", "LITIS". <br/>
--seed:           Seed to use for selection of dev set examples. Default is 151. <br/>
--version:        Version string for rerunning cache saving but without cache loading. Default is 'v0'. <br/>
--random_targets: Boolean to use random class pair or not. Default is True. <br/>
--classes:        Classes pair to evaluate on. Default is None which means random_targets must be used. <br/>

Valid classes per dataset: <br/>

    ESC-10: 'chainsaw', 'clock_tick', 'crackling_fire', 'crying_baby', 'dog',
          'helicopter', 'rain', 'rooster', 'sea_waves', 'sneezing'

    ESC-50: 'airplane', 'breathing', 'brushing_teeth', 'can_opening',
          'car_horn', 'cat', 'chainsaw', 'chirping_birds', 'church_bells',
          'clapping', 'clock_alarm', 'clock_tick', 'coughing', 'cow',
          'crackling_fire', 'crickets', 'crow', 'crying_baby', 'dog',
          'door_wood_creaks', 'door_wood_knock', 'drinking_sipping',
          'engine', 'fireworks', 'footsteps', 'frog', 'glass_breaking',
          'hand_saw', 'helicopter', 'hen', 'insects', 'keyboard_typing',
          'laughing', 'mouse_click', 'pig', 'pouring_water', 'rain',
          'rooster', 'sea_waves', 'sheep', 'siren', 'sneezing', 'snoring',
          'thunderstorm', 'toilet_flush', 'train', 'vacuum_cleaner',
          'washing_machine', 'water_drops', 'wind'

    UrbandSound8K: 'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
                  'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren',
                  'street_music'

    TUT-UrbanAcousticScenes: 'airport', 'bus', 'metro', 'metro_station', 'park',
                           'public_square', 'shopping_mall', 'street_pedestrian',
                           'street_traffic', 'tram'

    LITIS: 'avion', 'bus', 'busystreet', 'cafe', 'car', 'hall', 'hallgare',
         'kidgame', 'market', 'metro-paris', 'metro-rouen', 'poolhall',
         'quietstreet', 'restaurant', 'ruepietonne', 'shop', 'train-ter',
         'train-tgv', 'tubestation'

Experiments were performed with the following class pairs per dataset: <br/>

    ESC-10: [['clock_tick', 'crackling_fire'], ['crackling_fire', 'helicopter'], ['clock_tick', 'rain'], ['chainsaw', 'rooster'], ['crying_baby', 'helicopter'], <br/>
            ['rain', 'rooster'], ['clock_tick', 'sea_waves'], ['dog', 'rain'], ['crackling_fire', 'rooster'], ['helicopter', 'sea_waves']] <br/>

    ESC-50: [['chirping_birds', 'footsteps'], ['airplane', 'hen'], ['siren', 'sneezing'], ['breathing', 'keyboard_typing'], ['vacuum_cleaner', 'water_drops'], <br/>
             ['door_wood_creaks', 'rooster'], ['cow', 'crickets'], ['clapping', 'pouring_water'], ['crackling_fire', 'fireworks'], ['coughing', 'keyboard_typing']] <br/>

    UrbandSound8K: [['jackhammer', 'siren'], ['air_conditioner', 'children_playing'], ['drilling', 'children_playing'], ['jackhammer', 'children_playing'], <br/>
                    ['jackhammer', 'street_music'], ['siren', 'air_conditioner'], ['air_conditioner', 'dog_bark'], ['drilling', 'gun_shot'], <br/>
                    ['car_horn', 'siren']] <br/>

    TUT-UrbanAcousticScenes: [['park', 'metro_station'], ['bus', 'tram'], ['street_pedestrian', 'street_traffic'], ['street_pedestrian', 'bus'], <br/>
                              ['metro', 'shopping_mall'], ['street_pedestrian', 'park'], ['shopping_mall', 'street_traffic'], <br/>
                              ['shopping_mall', 'public_square'], ['airport', 'public_square'],  ['metro_station', 'street_pedestrian']] <br/>

    LITIS: [['bus', 'quietstreet'], ['cafe', 'shop'], ['quietstreet', 'tubestation'], <br/>
            ['hallgare', 'kidgame'], ['hall', 'shop'], ['bus', 'cafe'], ['hallgare', 'train-ter'], <br/>
            ['hallgare', 'restaurant'], ['car', 'market']] <br/>

Example commands: <br/>
All experiments were run with version tag: "v7". Use this tag if you wish to use the cache files provided. <br/>
```bash
python run_audio.py --model_name "vggish" --layer_idx_list "2" "5" "10" "15" --dataset_name "ESC-10" --num_prototypes 5 --dev_set_size 5 --classes "chainsaw" "crackling_fire" --seed 1 --cache 1 --version "v7"
```
```bash
python run_audio.py --model_name "vggish_svm" --layer_idx_list "21" --dataset_name "ESC-10" --dev_set_size 5 --classes "chainsaw" "crackling_fire" --seed 1 --cache 1 --version "v7"
```
```bash
python run_audio.py --model_name "soundnet" --layer_idx_list "3" "7" "17" --dataset_name "ESC-10" --num_prototypes 5 --dev_set_size 5 --classes "chainsaw" "crackling_fire" --seed 1 --cache 1 --version "v7"
```
```bash
python run_audio.py --model_name "soundnet_svm" --layer_idx_list "17" --dataset_name "ESC-10" --dev_set_size 5 --classes "chainsaw" "crackling_fire" --seed 1 --cache 1 --version "v7"
```
