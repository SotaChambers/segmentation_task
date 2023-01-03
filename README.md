# Segmentation Training

## Built Environment
- For GPU machine
```
bash docker/setup.sh
```

- For other machine
```
poetry install
poetry shell
```

- Download dataset
1 or 2
    1. Kaggle: [Find the nuclei in divergent images to advance medical discovery](https://www.kaggle.com/c/data-science-bowl-2018/data)
    2. using dvc
```
dvc pull
``` 

- Install Sphinx
```
poetry add "sphinx>=5.3.0"
poetry add sphinx_rtd_theme
```

- setting
```
sphinx-quickstart docs
> y と ja に設定
sphinx-apidoc -f -o docs/source/ ./model
sphinx-apidoc -f -o docs/source/ ./dataloader
sphinx-build docs/source docs/build
```
