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
