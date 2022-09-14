# StyleGAN with PluGeN

Code is based on [StyleFlow](https://github.com/RameenAbdal/StyleFlow). 



## Setup
#### Environment
```
conda env create -f environment.yml
```
#### Dataset
StyleGAN latents and attributes from StyleFlow ([download here](https://drive.google.com/file/d/1Kesr-oQ2XgXd6KZDxJ8Uy04xXZlbBZ0y/view?usp=sharing)).

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

## License
All rights reserved. Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (**Attribution-NonCommercial-ShareAlike 4.0 International**). The code is released for academic research use only.
