# PluGeN for molecular generation

In this repository, we present PluGeN model for molecular generation.

Our model works in two steps:
  1. Train the CharVAE model based on SMILES representation of molecules, that will be further used by PluGeN. The code is in `train_char_vae` directory.
  2. Train PluGeN for CharVAE, that could be used to generate molecules with given properties. The code is in `train_plugen` directory.

## Dependencies
Dependencies are in requirements.txt file. Use conda or python virual env to install them.

Installing by conda:
```
conda create --name <env_name> --file requirements.txt
```

Installing by virtual environment:
```
virtualenv env --python=python3
. env/bin/activate
pip install -r requirements.txt
```

