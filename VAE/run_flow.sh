# Phase 1
python train_CelebA.py -pg -mn "train_semisuper_paper_phase1" -t -mt Flow --flow-det const --train-mode vae_only  --epochs 100 --latent-sampling --flow-n-layers 8 --flow-hidden-dim 512 

# Phase 2
fhd=256
fnl=4
sdb=0.9
python train_CelebA.py --fully-supervised -pg -mn "train_fullsuper_phase2" --balanced-prior -t -mt Flow --flow-det const --train-mode flow_only  --epochs 80 --flow-n-layers $fnl --flow-hidden-dim $fhd --sigma 0.5 --sigma_decay_base $sdb --load train_phase1/model_save/ --latent-sampling &
