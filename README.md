# Harvard-Wayfair NLP

This repo contains public source code and demo notebooks for a data science capstone project
at Harvard (Fall 2020). The project aimed to help Wayfair, a major online retailer selling
furniture and home goods, to examine whether free-text reviews contain signals that improve
the prediction of return rates. Methods and findings of the project are summarized in
[this](https://towardsdatascience.com/read-the-reviews-analyzing-nlp-signals-of-wayfair-products-7c31b63cd369) blog post.

## Team Members

* Sangyoon Park
* Alex Spiride
* Erin Yang
* Jason Zhang

## Directory Structure
The repo contains two subdirectories: `src/` and `demo/`. `src/` contains source code for classes and functions to perform feature extraction and modeling. Building on this source code, `demo/` uses toy (fake) data to demonstrate how to perform feature extraction and modeling. We recommend you start with demo notebooks in `demo/notebooks/`.
```
├── README.md          <- The top-level README for developers using this project.
│
├── pyproject.toml     <- The file that defines build system, project metadata, and other
│                         install requirements.
│
├── poetry.lock        <- The file that resolves and downloads dependencies in `pyproject.toml`.
│
├── src
│   ├── features       <- Defines classes that can be used to extract NLP features.
│   └── models         <- Defines classes that can be used to predict product return rates.
│
├── demo               <- Materials showing how to use major resources in the repo.
│   ├── data
│   ├── models
│   └── notebooks
```

## Getting Started
The current project repo uses [`poetry`](https://python-poetry.org/docs/) to manage
dependencies among different Python packages, which is essential to reproducibility.
Following are steps for setting up and getting started:

First, ensure you are using the right version of Python (`^3.8`). We recommend you
use [`pyenv`](https://github.com/pyenv/pyenv) to effectively manage multiple versions
of Python installation. You can then install `poetry`:
```
$ pip install poetry
```

Once you clone the current repo into your local machine, you can go inside the repo and run:
```
$ poetry install
```
to install the right versions of packages for running scripts in the project repo.

To use the new Python configuration that has been installed, you need to run:
```
$ poetry shell
```
which will activate the virtual environment for the project repo.

You can simply type:
```
$ exit
```
to exit from the virtual environment and return to the global (or system) Python installation.

Once you set up the virtual environment using `poetry`, you can create the corresponding `jupyter` kernel as follows:
```
$ poetry run ipython kernel install --user --name=[desired-kernel-name]
```
Running a notebook on this new kernel (`[desired-kernel-name]`) will enable you to use the project-specific packages installed in the virtual environment.
