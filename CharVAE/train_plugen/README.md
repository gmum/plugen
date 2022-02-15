#  SMILES PluGeN

This code trains a PluGeN model for CharVAE, that could be used to modify the properties of molecules.


## Configure
We use gin-config to create simple training configuration.
The sample config file is in 'configs/config.gin'.
You can copy it and change model hyperparameters, training settings or training dataset.


## Run training
To train a PluGeN model for CharVAE simply run
```
python train.py --config_file 'path_to_config'
```

## Examples
In examples directory we included sample jupyter notebook with plugen-based molecular generation.
