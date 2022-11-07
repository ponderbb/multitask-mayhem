Multi-task learning for robust autonomy in mobile robots
==============================

A short description of the project.

Installation
------------
>[!Warning] I did not have too much time for this yet, please consult the Makefile!

Start by automatcially generating an environment for *conda* or *venv* with the following command in the top directory:
```shell
$ make create_environment
``` 
After activating this environment, install the requirements:
```shell
$ make requirements
```
Install pre-commit hooks for formatting tests, defined in `.pre-commit-config.yaml`:
```shell
$ pre-commit install
```

If you want to use the VSCode debugger with the virtual environment, append the following to the `launch.json` profile.

```shell
$ "python": "${command:python.interpreterPath}"
```


User guide
------------
### Unpacking the bagfiles from `/data/external` to `/data/raw/`

Use `-f True` if you want to overwrite already extracted bags in `data/raw`, otherwise they are skipped.

```shell
$ python src/data/rosbag_unpack.py
```
### Pruning the images from `/data/raw` to `/data/interim`

Similarity limit accoring to the SSIM measure (read more [here](https://pyimagesearch.com/2014/09/15/python-compare-two-images/)). The limit can be adjusted by passing `-s <float>`.

It will skip folders that has been already pruned and have an existing folder in the target directory `/data/interim`. This can be bypassed with `-f`. 

Individual bag can be extracted by passing `-b -i <path_to_bag>`, which assumes that the output folder can be overwritten, if it exists.

Passing `-d` will enable debug level logging information and limit the loop to the last bag in the list.

```shell
$ python src/data/filter_images.py -s 0.7
```

Issues
---

In case of `ModuleNotFoundError: No module named 'src'`, you should append the following to the end of the environment activation file and restart the virtual environment.

```shell
$ echo 'export PYTHONPATH=$PYTHONPATH:$(pwd)' >> ~/.virtualenvs/multitask-mayhem/bin/activate
```
or
```shell
$ conda install conda-build
$ conda develop . 
```


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
