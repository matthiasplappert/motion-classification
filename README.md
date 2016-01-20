This code was written during the course of my Bachelor Thesis "Classification of Human Whole-Body Motion using Hidden Markov Models".
Some things might be broken and I definitely don't recommend to use any of the code in any sort of production application.
However, for research purposes this code might be useful so I decided to open-source it.
Use at your own risk!

# Requirements
Use pip to install most requriements (`pip install -r requriements.txt`). Sometimes this causes
problems if Cython, numpy and scipy are not already installed, in which case this needs to be done
manually.

Additionally, some packages must be installed that are not provided by pip.

## pySimox and pyMMM
pySimox and pyMMM must be installed manually as well. To build them, perform the following steps:

```
git submodule update --init --recursive
cd vendor/pySimox/build
cmake ..
make
cp _pysimox.so ../../../lib/python2.7/site-packages/_pysimox.so
cp pysimox.py ../../../lib/python2.7/site-packages/pysimox.py
cd ../pyMMM/build
cmake ..
make
cp _pymmm.so ../../../lib/python2.7/site-packages/_pymmm.so
cp pymmm.py ../../../lib/python2.7/site-packages/pymmm.py
```
Note that the installation script may need some fine-tuning. Additionally, this assumes that all
`virtualenv` is set up in the root of this git repo.

# Basic Usage
This repo contains two main programs: `dataset.py` and `evaluate_new.py`.
All of them are located in `src` and should be run from this directory. There are some additional
files in there, some of them are out-dated and should be deleted (e.g. `evaluate.py`), some of
them are really just scripts and should be moved to the `scripts` folder eventually.

## The `dataset` tool
The `dataset` tool is concerened
with handling everything related to datasets: `plot` plots features, `export` saves a dataset in a variety
of formats, `report` prints details about a dataset and `check` performs a consistency check. Additionally,
`export-all` can be used to create a dataset that contains all features (normalized and unnormalized)
by merging Vicon C3D and MMM files into one giant file. A couple of examples:

- `python dataset.py ../data/dataset1.json plot --features root_pos` plots the `root_pos` feature of all motions in the dataset; the dataset can be a JSON
manifest or a pickled dataset
- `python dataset.py ../data/dataset1.json export --output ~/export.pkl` exports dataset1 as a single pickled file; usually a JSON manifest is used
- `python dataset.py ../data/dataset1.json export-all --output ~/export_all.pkl` exports dataset1 by combining vicon and MMM files and by computing
both the normalized and unnormalized version of all features. It also performs normalization on the vicon data by using
additional information from the MMM data (namely the root_pos and root_rot); the dataset has to be a JSON manifest
- `python dataset.py ../data/dataset1.json report` prints details about a dataset; the dataset can be a JSON
manifest or a pickled dataset
- `python dataset.py ../data/dataset1.json check` performs a consistency check of a dataset; the manifest has to be a JSON
manifest

Additional parameters are avaialble for most commands. Use `dataset --help` to get an overview.

## The `evaluate_new` tool
The `evaluate_new` tool can be used to perform feature selection (using the `feature` command)
or to evaluate different types of models with decision makers (by using the `model` command). It
is important to note that the `evaluate_new` tool expects a pickled version of the dataset, hence `export`
or `export_all` must be used to prepare a dataset. This is to avoid the computational complexity.

A couple of examples:
- `python evaluate_new.py model ../data/export_all.pkl --features normalized_joint_pos normalized_root_pos --decision-maker log-regression --n-states 5 --model fhmm-seq --output-dir ~/out` trains
a HMM ensemble with each HMM having 5 states on the normalized_joint_pos and normalized_root_pos features and uses logistic regression to perform the final predicition. The results are also
saved in the directory ~/out
- `python evaluate_new.py features ../data/export_all.pkl --features normalized_joint_pos normalized_root_pos --measure wasserstein` performs feature selection using the starting
set normalized_joint_pos normalized_root_pos and the wasserstein measure

## From dataset to result
First, define a JSON manifest `dataset.json` that links together the individual motions and pick labels. Next, export
the dataset by using `python dataset.py ../data/dataset.json export-all --output ../data/dataset_all.pkl`. If you need
smoothing, simply load the dataset (using `pickle.load()`), call `smooth_features()` on the `Dataset` object and dump it to a
new file. There's currently no script for this but it can be done using three lines and the interactive python interpreter.
Next, perform feature selection using `python evaluate_new.py features ../data/dataset_all.pkl --features <list of features> --measure wasserstein --output-dir ~/features --transformers minmax-scaler`. You'll want to use the minmax scaler transformer to avoid numerical problems during training.
This will probably take a while. The results (at ~/features) will give you the best feature subsets that were found.
Next, use those features to train an HMM ensemble: `python evaluate_new model ../data/dataset_all.pkl --features <best features> --model fhmm-seq --n-chains 2 --n-states 10 --n-training-iter 30 -decision-maker log-regression --transformers minmax-scaler --output-dir ~/train` (again, the minmax-scaler is almost always a good idea). The results will be
in ~/output.
