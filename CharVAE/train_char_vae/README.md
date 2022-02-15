# CharVAE

This code trains a CharVAE model based on SMILES representation of molecules, that could be further used by PluGeN.


## Configure
We use gin-config to create simple training configuration.
The sample config file is in 'configs/config.gin'.
You can copy it and change model hyperparameters, training settings or training dataset.


## Run training
To train the CharVAE model simply run
```
python train.py --config_file 'path_to_config'
```
