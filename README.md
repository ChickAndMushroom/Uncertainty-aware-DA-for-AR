# Uncertainty-Aware Domain Adaptation for Action Recognition


---
This is the official PyTorch implementation of our papers:

**Uncertainty-Aware Domain Adaptation for Action Recognition**  

ABSTRACT
---
## Contents
* [Requirements](#requirements)
* [Dataset Preparation](#dataset-preparation)
  * [Data structure](#data-structure)
  * [File lists for training/validation](#file-lists-for-trainingvalidation)
  * [Input data](#input-data)
* [Usage](#usage)
  * [Training](#training)
  * [Testing](#testing)
<!--   * [Video Demo](#video-demo) -->
* [Options](#options)
  * [Domain Adaptation](#domain-adaptation)
  * [More options](#more-options)
* [Citation](#citation)
* [Contact](#contact)

---
## Requirements
* support Python 3.6, PyTorch 0.4, CUDA 9.0, CUDNN 7.1.4
* install all the library with: `pip install -r requirements.txt`

---
## Dataset Preparation
### Data structure
You need to extract frame-level features for each video to run the codes. To extract features, please check [`dataset_preparation/`](dataset_preparation/).

Folder Structure:
```
DATA_PATH/
  DATASET/
    list_DATASET_SUFFIX.txt
    RGB/
      CLASS_01/
        VIDEO_0001.mp4
        VIDEO_0002.mp4
        ...
      CLASS_02/
      ...

    RGB-Feature/
      VIDEO_0001/
        img_00001.t7
        img_00002.t7
        ...
      VIDEO_0002/
      ...
```
`RGB-Feature/` contains all the feature vectors for training/testing. `RGB/` contains all the raw videos.

There should be at least two `DATASET` folders: source training set  and validation set. If you want to do domain adaption, you need to have another `DATASET`: target training set.

### File lists for training/validation
The file list `list_DATASET_SUFFIX.txt` is required for data feeding. Each line in the list contains the full path of the video folder, video frame number, and video class index. It looks like:
```
DATA_PATH/DATASET/RGB-Feature/VIDEO_0001/ 100 0
DATA_PATH/DATASET/RGB-Feature/VIDEO_0002/ 150 1
......
```
To generate the file list, please check [`dataset_preparation/`](dataset_preparation/).

### Input data
Here we provide pre-extracted features and data list files, so you can skip the above two steps and directly try our training/testing codes. You may need to manually edit the path in the data list files.
* Features
  * UCF: [download](https://www.dropbox.com/s/ebtc1hz1paz9bmz/ucf101-feat.zip?dl=0)
  * HMDB: [download](https://www.dropbox.com/s/aiac0ytb9jt83a2/hmdb51-feat.zip?dl=0)
  * Olympic: [training](https://www.dropbox.com/s/0ljfsp52hydyqht/olympic_train-feat.zip?dl=0) | [validation](https://www.dropbox.com/s/yh09a2th4hf8hqp/olympic_val-feat.zip?dl=0)
* Data lists
  * UCF-Olympic
    * UCF: [training list](https://www.dropbox.com/s/du8d3qrzs9h8phn/list_ucf101_train_ucf_olympic-feature.txt?dl=0) | [validation list](https://www.dropbox.com/s/0qrhuen3o27g9k5/list_ucf101_val_ucf_olympic-feature.txt?dl=0)
    * Olympic: [training list](https://www.dropbox.com/s/0eafz1kjk71i0i9/list_olympic_train_ucf_olympic-feature.txt?dl=0) | [validation list](https://www.dropbox.com/s/ku27uniw4xm7wpm/list_olympic_val_ucf_olympic-feature.txt?dl=0)
  * UCF-HMDB<sub>small</sub>
    * UCF: [training list](https://www.dropbox.com/s/2g04infpxwysjfb/list_ucf101_train_hmdb_ucf_small-feature.txt?dl=0) | [validation list](https://www.dropbox.com/s/6fjour5n1dcabfy/list_ucf101_val_hmdb_ucf_small-feature.txt?dl=0)
    * HMDB: [training list](https://www.dropbox.com/s/q6e7jwhr1ktmrrt/list_hmdb51_train_hmdb_ucf_small-feature.txt?dl=0) | [validation list](https://www.dropbox.com/s/qh3h619bdo2q3h1/list_hmdb51_val_hmdb_ucf_small-feature.txt?dl=0)
  * UCF-HMDB<sub>full</sub>
    * UCF: [training list](https://www.dropbox.com/s/jrahoh6u8k90iec/list_ucf101_train_hmdb_ucf-feature.txt?dl=0) | [validation list](https://www.dropbox.com/s/7359sfsflfkf60c/list_ucf101_val_hmdb_ucf-feature.txt?dl=0)
    * HMDB: [training list](https://www.dropbox.com/s/thj7mkzof6pgfmj/list_hmdb51_train_hmdb_ucf-feature.txt?dl=0) | [validation list](https://www.dropbox.com/s/s9yc43u87kjcdhx/list_hmdb51_val_hmdb_ucf-feature.txt?dl=0)

* Kinetics-Gameplay: please fill [**this form**](https://forms.gle/bziHhvQJGmi7hwF26) to request the features and data lists. <br>
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>
The Kinetics-Gameplay dataset is licensed under <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a> for non-commercial purposes only.
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---
## Usage
* training/validation: Run `./script_train_val.sh`
<!-- * demo video: Run `./script_demo_video.sh` -->

All the commonly used variables/parameters have comments in the end of the line. Please check [Options](#options).

#### Training
All the outputs will be under the directory `exp_path`.
* Outputs:
  * model weights: `checkpoint.pth.tar`, `model_best.pth.tar`
  * log files: `train.log`, `train_short.log`, `val.log`, `val_short.log`

#### Testing
You can choose one of model_weights for testing. All the outputs will be under the directory `exp_path`.

* Outputs:
  * score_data: used to check the model output (`scores_XXX.npz`)
  * confusion matrix: `confusion_matrix_XXX.png` and `confusion_matrix_XXX-topK.txt`

<!-- #### Video Demo
`demo_video.py` overlays the predicted categories and confidence values on one video. Please see "Results". -->

---
## Options
#### Domain Adaptation
<!-- In both `./script_train_val.sh` and `./script_demo_video.sh`, there are several options related to our Domain Adaptation approaches. -->
In `./script_train_val.sh`, there are several options related to our DA approaches.
* `use_target`: switch on/off the DA mode
  * `none`: not use target data (no DA)
  * `uSv`/`Sv`: use target data in a unsupervised/supervised way

<!-- * options for the DA approaches:
  * discrepancy-based: DAN, JAN
  * adversarial-based: RevGrad
  * Normalization-based: AdaBN
  * Ensemble-based: MCD -->

#### More options
For more details of all the arguments, please check [opts.py](opts.py).

#### Notes
The options in the scripts have comments with the following types:
* no comment: user can still change it, but NOT recommend (may need to change the code or have different experimental results)
* comments with choices (e.g. `true | false`): can only choose from choices
* comments as `depend on users`: totally depend on users (mostly related to data path)
