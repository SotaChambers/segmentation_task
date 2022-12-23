# Segmentation Training

## Built Environment
- create `.gitignore`
```shell
curl -L http://www.gitignore.io/api/python,windows,osx
```

- install package
```
poetry add segmentation-models-pytorch
albumentations,
flake8,
matplotlib
numpy,
opencv-contrib-python
```

- flake8
```
mkdir .config
touch .config/flake8
```

- Download dataset
Kaggle: [Find the nuclei in divergent images to advance medical discovery](https://www.kaggle.com/c/data-science-bowl-2018/data)
