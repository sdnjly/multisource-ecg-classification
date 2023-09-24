# Learning with Incomplete Labels of Multisource Datasets for ECG Classification

This repository contains the source code and associated files for the research study titled "Learning with Incomplete Labels of Multisource Datasets for ECG Classification".

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)


## Introduction

This research study focuses on addressing the challenge of classifying electrocardiogram (ECG) data from multiple sources with incomplete labels. The study proposes a deep-learning-based framework to improve the accuracy of ECG classification in such scenarios. This repository contains the source code and resources related to the research.

## Getting Started

These instructions will help you get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before you begin, ensure you have met the following requirements:
- [Python](https://www.python.org/downloads/) (>=3.6)
- [TensorFlow](https://www.tensorflow.org/install) (>=2.0)
- Other dependencies (listed in `requirements.txt`)

### Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/yourusername/multisource-ecg-classification.git
   cd multisource-ecg-classification
   ```

2. Create a virtual environment (recommended):

	```shell
	python -m venv venv
	source venv/bin/activate
	```
	
3. Install required packages:

	```shell
	pip install -r requirements.txt
	```
	
### Data Preparation

The datasets used in this study can be found from the Physionet/CinC challenge 2020/2021 websites. You can follow instructions on the websites to downloads the datasets.

- [Physionet/CinC challenge 2020](https://moody-challenge.physionet.org/2020/)
- [Physionet/CinC challenge 2021](https://moody-challenge.physionet.org/2021/)
	
### Usage	

1. model training

	```shell
    python train_model.py training_data model
	```
	
	`training_data` is a folder of training data files, `model` is a folder for saving your models, `test_data` is a folder of test data files
	
2. model testing
	
	```shell
    python test_model.py model test_data test_outputs
	```

	