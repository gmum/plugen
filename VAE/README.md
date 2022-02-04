# PluGeN with VAE backbone
VAE-based experiments on CelebA dataset

This code is based on [MSP](https://github.com/lissomx/MSP).

Download the CelebA dataset and put the `img_align_celeba.zip` file with images and the `list_attr_celeba.txt` file with labels in the `./CelebA_Dataset/` directory. Run `bash run_flow.sh` to train the model. This command will first train the baseline VAE and then PluGeN on top of that. 