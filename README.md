# Hidden_adversarial attacks

We propose concealed adversarial attacks for different time-series models that are hard to detect visually or via a specially trained discriminator. 
To achieve this goal, we train a discriminator model to detect adversarial perturbation in data. This model allows to define an additional regularization for an attack. 
The proposed adversarial attack with the regularization makes attacks harder to detect. Our claims are supported by extensive benchmarking via datasets from UCR and different models: recurrent, convolution, state-space, and transformers. For our proposed metric, a mix of concealability and effectiveness, we find an improvement over vanilla attacks. In some cases, the findings are more intricate, as adversarial vulnerability significantly changes for different models and datasets. Thus, our proposed attack can bypass the discriminator, raising the challenge of creating neural network models for time-series data that can resist adversarial attacks.

## Data
 Datasets that we used are available [online](https://disk.yandex.ru/d/--KIPMEJ-cB4MA). Folder `raw/` contains the datasets in their initial formats, and the final .parquet files we use are stored in the `preprocessed/` folder. To reproduce the experiments, download the `preprocessed/` folder and put it into the `data/` directory.

## Quick Start

If you want to work with project with Docker, you can use folder **docker_scripts**.
Firstly copy file `credentials_example` to `credentials` and tune it with your variables for running docker. After you need to make docker image using command:

```
cd docker_scripts
bash build
```

For creating container run:

```
bash launch_container
```

All the requirements are listed in `requirements.txt`
For install all packages run

```
pip install -r requirements.txt
```

After you need to create folders `checkpoints` for saving classifier weights and `results` for saving adversarial attacks results.

Where are three basic steps: train classifier, attack model, train discriminator.
To run these steps you need to rename folder `config_examples` to `config`, then change assosiated config files in "config" folder and after that run assosiated python scrits `train_classifier.py`, `attack_run.py` and `train_discriminator.py`.

For example:
```
python train_classifier.py
```
## Describtion

The goal of the project is to create hardly detected adversarial attacks for time-series models.

## Content

| File or Folder | Content |
| --- | --- |
| checkpoints| folders for saving weights of the models |
| config | folder contains config files with params of models and paths |
| data | folder contains datasets For LSTM model (Ford_A) and datasets UCR, UEA datasets for TS2Vec models |
| docker_scripts | folder for set environment in docker container (if needed) |
| notebooks | folder with notebooks for data visualisation and small experiments|
| src | folder with code|

## Configs structure
We use a hydra-based experiment configuration structure. To run `train_discriminator.py` and `attack_run.py` you need to set the following configs:
* `- dataset:` choose the dataset you need from the `config/dataset/` directory.
* `- model:` choose the model for discriminator from the `config/model/` directory.
* `- model@attack_model:` choose the classifier model from the `config/model/` directory which was trained using a script `train_classifier.py`.
* `- model@disc_model_check:` select the model for discriminator from the `config/model/` directory if you want to get metrics of another  trained discriminator. Note that if you want to use it, set `use_disc_check: True` 
* `- attack_disc@attack:` choose the type of adversarial attack from the `config/attack/` directory.

