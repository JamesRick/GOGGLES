## Installation

```bash
git clone https://github.com/JamesRick/GOGGLES
git checkout audio
git submodule update --init --recursive
cd GOGGLES
pip install -e .
```

## Data Files
The datasets used can be downloaded from the following links:

ESC-10: https://drive.google.com/file/d/1DyUH55fcSYfzSLjxMAAujXSbT6Gbujdc/view?usp=sharing
ESC-50: https://drive.google.com/file/d/1mkOOQ0WakozbAKsjhPeMJaTtRng2kgtn/view?usp=sharing
UrbanSound8K: https://drive.google.com/file/d/1U-lXuzkLSzVN1H_y9bnZdqa0omQa72kW/view?usp=sharing
TUT-UrbanAcousticScenes: https://drive.google.com/file/d/1qe1gR3B77H_ys52VHXRpJNAD432pxbEe/view?usp=sharing
LITIS: https://drive.google.com/file/d/19fiBtGrZ_IVWcRdb7-zrrW7-L3Q65TxT/view?usp=sharing

ESC-10 is the smallest dataset, so this download will be the fastest.
Download the {dataset-name}.tar.gz file and extract it into the GOGGLES/goggles/data/ folder.

IMPORTANT NOTE: Datasets are expected to have the directory names as shown above and be located in the GOGGLES/goggles/data folder.

### Examples
tar -xzf {dataset-name}.tar.gz GOGGLES/goggles/data/{dataset-name}

## VGGish weights
The weights for the vggish model are stored in the following link:
VGGish Weights: https://drive.google.com/file/d/1jsFol7qtN1Nsq3vCPF_rVlyV5OmtugLG/view?usp=sharing

After completing the download place the pytorch_vggish.pth file in the GOGGLES/goggles/torch_vggish directory.

IMPORTANT NOTE: These weights are required for running the vggish model and must be placed in the directory as specified above.

## Results and output files
Results are written to pickle files in the GOGGLES/goggles/output directory. These pickle files are parsed with GOGGLES/goggles/utils/postprocessing.py and converted to a csv located in GOGGLES/goggles/results/full_results.csv

Pickle files and full_results csv for the completed experiments can be downloaded here:

Output Directory: https://drive.google.com/file/d/1gvgjfsPBzm-Fq6AJ_hWAVdYWiwzycThr/view?usp=sharing
Results Directory: https://drive.google.com/drive/folders/12sW_mwsCp11diny2P3z-Q2px97UXIXCt?usp=sharing

Both of these directories should be placed into the goggles directory

## \_Scratch directory
I've also supplied the cached affinity functions for every run of the experiments in the following link:
\_scratch: https://drive.google.com/file/d/1DBabZ4KUwdRmyL3dVdB3fNxI2e8kZieD/view?usp=sharing

Extract this to the GOGGLES/goggles/ directory if you wish to use these cache files.

## Example Usage
Main run script is GOGGLES/goggles/test/run_audio.py
Default parameters will run SoundNet with ESC-10 dataset.

Parameters:
--layer_idx_list: The list of max pooling layers to gather affinity functions from. [3, 7, 17] for SoundNet. [2, 5, 10, 15] for VGGish.
--num_prototypes: Number of prototypes per layer for affinity functions. Default is 10
--dev_set_size:   Size of the development set to use for cluster to class inference. Default is 5.
--model_name:     Name of the model to use. Options: "vggish", "soundnet", "soundnet_svm", "vggish_svm"
--cache:          Boolean to use cache or not. Default is false, so omit this parameter if you do not wish to use the cache.
--dataset_name:   Name of the dataset to use. Default is "ESC-10". Options: "ESC-10", "ESC-50", "UrbanSound8K", "TUT-UrbanAcousticScenes", "LITIS"
--seed:           Seed to use for selection of dev set examples. Default is 151.
--version:        Version string for rerunning cache saving but without cache loading. Default is 'v0'
--random_targets: Boolean to use random class pair or not. Default is True.
--classes:        Classes pair to evaluate on. Default is None which means random_targets must be used.
                  Valid classes per dataset:
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

Experiments were performed with the following class pairs per dataset:
ESC-10: [['clock_tick', 'helicopter'], ['dog', 'sneezing'], ['chainsaw', 'crackling_fire'], ['rooster', 'dog'], ['clock_tick', 'rain'],
        ['rooster', 'rain'], ['helicopter', 'sneezing'], ['clock_tick', 'dog']]

ESC-50: [['insects', 'hand_saw'], ['crickets', 'cow'], ['pouring_water', 'clapping'], ['fireworks', 'crackling_fire'], ['coughing', 'keyboard_typing'],
         ['glass_breaking', 'rooster'], ['door_wood_knock', 'engine'], ['can_opening', 'pig'], ['water_drops', 'glass_breaking'], ['snoring', 'brushing_teeth']]

UrbandSound8K: [['jackhammer', 'siren'], ['air_conditioner', 'children_playing'], ['drilling', 'children_playing'], ['jackhammer', 'children_playing'],
                ['jackhammer', 'street_music'], ['siren', 'air_conditioner'], ['air_conditioner', 'dog_bark'], ['drilling', 'gun_shot'],
                ['car_horn', 'siren']]

TUT-UrbanAcousticScenes: [['park', 'metro_station'], ['bus', 'tram'], ['street_pedestrian', 'street_traffic'], ['street_pedestrian', 'bus'],
                          ['metro', 'shopping_mall'], ['street_pedestrian', 'park'], ['shopping_mall', 'street_traffic'],
                          ['shopping_mall', 'public_square'], ['airport', 'public_square'],  ['metro_station', 'street_pedestrian']]

LITIS: [['quietstreet', 'kidgame'], ['car', 'ruepietonne'], ['tubestation', 'train-tgv'],
        ['poolhall', 'metro-rouen'], ['restaurant', 'metro-paris'], ['car', 'shop'], ['restaurant', 'cafe'],
        ['train-ter', 'busystreet'], ['market', 'train-tgv']]

Example commands:
All experiments were run with version tag: "v7". Use this tag if you wish to use the cache files provided.
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
