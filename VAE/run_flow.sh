# Phase 1
python train_CelebA.py -pg -mn "train_phase1" -t -mt Flow --flow-det const --train-mode vae_only  --epochs 50 --latent-sampling --flow-n-layers 4 --flow-hidden-dim 256

# Phase 2
python train_CelebA.py --fully-supervised -pg -mn "train_phase2" --balanced-prior -t -mt Flow --flow-det const --train-mode flow_only  --epochs 100 --flow-n-layers 4 --flow-hidden-dim 256 --sigma 0.7 --sigma_decay_base 0.9 --load train_phase1/model_save/ --latent-sampling
