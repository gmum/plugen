# StyleGAN with PluGeN

Code is based on [StyleFlow](https://github.com/RameenAbdal/StyleFlow).

## Setup
#### Environment
```
conda env create -f environment.yml
```
#### Dataset
StyleGAN latents and attributes ([download here](https://drive.google.com/file/d/1opdzeqpYWtE1uexO49JI-3_RWfE9MYlN/view)).

## Training
#### PluGeN
```
python train.py
```
#### StyleFlow
```
python train_styleflow.py
```

### Evaluate
#### PluGeN
```
python evaluate.py
```
#### StyleFlow
```
python evaluate.py --styleflow
```
