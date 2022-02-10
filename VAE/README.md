# PluGeN with VAE backbone
VAE-based experiments on CelebA dataset.

This codebase is based on [MSP](https://github.com/lissomx/MSP).

## Setup
Setup the environment using conda:
```
conda env create -f environment.yml
```

Then Download the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and put the `img_align_celeba.zip` file with images and the `list_attr_celeba.txt` file with labels in the `./CelebA_Dataset/` directory.

## Training
Run 
```
bash run_flow.sh
```
This command will first train the baseline VAE and then PluGeN on top of that. Examples of generated and modified samples will be placed in the `train_phase2/Outputs` directory and the weights in the `train_phase2/model_save` directory.
