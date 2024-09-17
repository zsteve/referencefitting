import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import time
from os import environ
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from bicycle.callbacks import (
    CustomModelCheckpoint,
    GenerateCallback,
    MyLoggerCallback,
)
from bicycle.dictlogger import DictLogger
from bicycle.model import BICYCLE
from bicycle.utils.data import (
    compute_inits,
    create_data,
    create_loaders,
    get_diagonal_mask,
)
from bicycle.utils.general import get_full_name
from bicycle.utils.plotting import plot_training_results
from pytorch_lightning.callbacks import RichProgressBar, StochasticWeightAveraging
from pytorch_lightning.tuner.tuning import Tuner

n_factors = 0
add_covariates = False
n_covariates = 0  # Number of covariates
covariate_strength = 5.0
correct_covariates = False

intervention_type_simulation = "Cas9"
intervention_type_inference = "Cas9"

SEED = 1
pl.seed_everything(SEED)
torch.set_float32_matmul_precision("high")
device = torch.device("cpu")
user_dir = "."
MODEL_PATH = Path(os.path.join(user_dir, "models"))
PLOT_PATH = Path(os.path.join(user_dir, "plots"))
MODEL_PATH.mkdir(parents=True, exist_ok=True)
PLOT_PATH.mkdir(parents=True, exist_ok=True)

import importlib
import os
import sys
sys.path.append("../src/")
import util
importlib.reload(util)
import pandas as pd
import glob
s = sys.argv[1]
rep = sys.argv[2]
DATA_PATH = sys.argv[3]
paths = glob.glob(os.path.join(DATA_PATH, f"{s}/{s}*-{rep}")) + glob.glob(os.path.join(DATA_PATH, f"{s}_ko*/{s}*-{rep}"))
names = [os.path.basename(p).split("-")[1] for p in paths]
adatas = [util.load_adata(p, log_transform = False) for p in paths]

kos = []
for p in paths:
    try:
        kos.append(os.path.basename(p).split('_ko_')[1].split("-")[0])
    except:
        kos.append(None)

samples = torch.tensor(np.vstack([(25 * a.X).astype(int) for a in adatas]))
genes = adatas[0].var.index
intervened_variables = torch.tensor(np.vstack([np.tile((genes == k).astype(int), (a.shape[0], 1)) for (a, k) in zip(adatas, kos)]))
gt_interv = torch.tensor(np.vstack([np.tile((genes == k).astype(int), (1, 1)) for (a, k) in zip(adatas, kos)]).T)
sim_regime = torch.tensor(np.hstack([np.full((a.shape[0], ), i) for (i, a) in enumerate(adatas)]))
beta = None
n_genes = len(genes)

NUM_ITERS=25_000
NUM_ITERS_PRETRAIN=5_000

# TRAINING
lr = 1e-3  # 3e-4
batch_size = 10_000
USE_INITS = False
use_encoder = False
n_epochs = NUM_ITERS
early_stopping = False
early_stopping_patience = 500
early_stopping_min_delta = 0.01
# Maybe this helps to stop the loss from growing late during training (see current version
# of Plot_Diagnostics.ipynb)
optimizer = "adam"  # "rmsprop" #"adam"
optimizer_kwargs = {"betas": [0.5, 0.9]}  # Faster decay for estimates of gradient and gradient squared
gradient_clip_val = 1.0
GPU_DEVICE = 0
plot_epoch_callback = 500
validation_size = 0.2
lyapunov_penalty = True
swa = 250
n_epochs_pretrain_latents = NUM_ITERS_PRETRAIN

rank_w_cov_factor = n_genes  # Same as dictys: #min(TFs, N_GENES-1)
perfect_interventions = True

# MODEL
x_distribution = "Multinomial"
x_distribution_kwargs = {}
model_T = 1.0
learn_T = False
use_latents = True

LOGO = []
train_gene_ko = [str(x) for x in set(range(0, n_genes)) - set(LOGO)]  # We start counting at 0
name_prefix = f"bicycle_BoolODE_{s}_{rep}"
SAVE_PLOT = True
CHECKPOINTING = False
VERBOSE_CHECKPOINTING = False
OVERWRITE = True
# REST
n_samples_total = samples.shape[0] # n_samples_control + (len(train_gene_ko) + len(test_gene_ko)) * n_samples_per_perturbation
check_val_every_n_epoch = 1
log_every_n_steps = 1
# FIXME: There might be duplicates...
ho_perturbations = sorted(
    list(set([tuple(sorted(np.random.choice(n_genes, 2, replace=False))) for _ in range(0, 20)]))
)
test_gene_ko = [f"{x[0]},{x[1]}" for x in ho_perturbations]

# Create Mask
mask = get_diagonal_mask(n_genes, device)

if n_factors > 0:
    mask = None

check_samples, check_gt_interv, check_sim_regime, check_beta = (
    np.copy(samples),
    np.copy(gt_interv),
    np.copy(sim_regime),
    np.copy(beta),
)


train_loader, validation_loader, test_loader = create_loaders(
    samples, sim_regime, validation_size, batch_size, SEED, train_gene_ko, test_gene_ko
)

covariates = None

# if USE_INITS:
#     init_tensors = compute_inits(train_loader.dataset, rank_w_cov_factor, n_contexts)

print("Training data:")
print(f"- Number of training samples: {len(train_loader.dataset)}")
if validation_size > 0:
    print(f"- Number of validation samples: {len(validation_loader.dataset)}")
if LOGO:
    print(f"- Number of test samples: {len(test_loader.dataset)}")

device = torch.device(f"cpu")
gt_interv = gt_interv.to(device)
n_genes = samples.shape[1]

if covariates is not None and correct_covariates:
    covariates = covariates.to(device)

for scale_kl in [1.0]:  # 1
    for scale_l1 in [0.1]:
        for scale_spectral in [0.0]:  # 1.0
            for scale_lyapunov in [1.0]:  # 0.1
                file_dir = get_full_name(
                    name_prefix,
                    len(LOGO),
                    SEED,
                    lr,
                    n_genes,
                    scale_l1,
                    scale_kl,
                    scale_spectral,
                    scale_lyapunov,
                    gradient_clip_val,
                    swa,
                )

                # If final plot or final model exists: do not overwrite by default
                print("Checking Model and Plot files...")
                final_file_name = os.path.join(MODEL_PATH, file_dir, "last.ckpt")
                final_plot_name = os.path.join(PLOT_PATH, file_dir, "last.png")

                # Save simulated data for inspection and debugging
                final_data_path = os.path.join(PLOT_PATH, file_dir)

                if os.path.isdir(final_data_path):
                    print(final_data_path, "exists")
                else:
                    print("Creating", final_data_path)
                    os.mkdir(final_data_path)

                np.save(os.path.join(final_data_path, "check_sim_samples.npy"), check_samples)
                np.save(os.path.join(final_data_path, "check_sim_regimes.npy"), check_sim_regime)
                # np.save(os.path.join(final_data_path, "check_sim_beta.npy"), check_beta)
                np.save(os.path.join(final_data_path, "check_sim_gt_interv.npy"), check_gt_interv)

                if (Path(final_file_name).exists() & SAVE_PLOT & ~OVERWRITE) | (
                    Path(final_plot_name).exists() & CHECKPOINTING & ~OVERWRITE
                ):
                    print("- Files already exists, skipping...")
                    continue
                else:
                    print("- Not all files exist, fitting model...")
                    print("  - Deleting dirs")
                    # Delete directories of files
                    if Path(final_file_name).exists():
                        print(f"  - Deleting {final_file_name}")
                        # Delete all files in os.path.join(MODEL_PATH, file_name)
                        for f in os.listdir(os.path.join(MODEL_PATH, file_dir)):
                            os.remove(os.path.join(MODEL_PATH, file_dir, f))
                    if Path(final_plot_name).exists():
                        print(f"  - Deleting {final_plot_name}")
                        for f in os.listdir(os.path.join(PLOT_PATH, file_dir)):
                            os.remove(os.path.join(PLOT_PATH, file_dir, f))

                    print("  - Creating dirs")
                    # Create directories
                    Path(os.path.join(MODEL_PATH, file_dir)).mkdir(parents=True, exist_ok=True)
                    Path(os.path.join(PLOT_PATH, file_dir)).mkdir(parents=True, exist_ok=True)

                model = BICYCLE(
                    lr,
                    gt_interv,
                    n_genes,
                    n_samples=n_samples_total,
                    lyapunov_penalty=lyapunov_penalty,
                    perfect_interventions=perfect_interventions,
                    rank_w_cov_factor=rank_w_cov_factor,
                    init_tensors=init_tensors if USE_INITS else None,
                    optimizer=optimizer,
                    optimizer_kwargs=optimizer_kwargs,
                    device=device,
                    scale_l1=scale_l1,
                    scale_lyapunov=scale_lyapunov,
                    scale_spectral=scale_spectral,
                    scale_kl=scale_kl,
                    early_stopping=early_stopping,
                    early_stopping_min_delta=early_stopping_min_delta,
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_p_mode=True,
                    x_distribution=x_distribution,
                    x_distribution_kwargs=x_distribution_kwargs,
                    mask=mask,
                    use_encoder=use_encoder,
                    gt_beta=beta,
                    train_gene_ko=train_gene_ko,
                    # test_gene_ko=test_gene_ko,
                    use_latents=use_latents,
                    covariates=covariates,
                    n_factors=n_factors,
                    intervention_type=intervention_type_inference,
                    T=model_T,
                    learn_T=learn_T,
                )
                model.to(device)

                dlogger = DictLogger()
                loggers = [dlogger]

                callbacks = [
                    RichProgressBar(refresh_rate=1),
                    GenerateCallback(
                        final_plot_name, plot_epoch_callback=plot_epoch_callback, 
                    ),
                ]
                if swa > 0:
                    callbacks.append(StochasticWeightAveraging(0.01, swa_epoch_start=swa))
                if CHECKPOINTING:
                    Path(os.path.join(MODEL_PATH, file_dir)).mkdir(parents=True, exist_ok=True)
                    callbacks.append(
                        CustomModelCheckpoint(
                            dirpath=os.path.join(MODEL_PATH, file_dir),
                            filename="{epoch}",
                            save_last=True,
                            save_top_k=1,
                            verbose=VERBOSE_CHECKPOINTING,
                            monitor="valid_loss",
                            mode="min",
                            save_weights_only=True,
                            start_after=0,
                            save_on_train_epoch_end=False,
                            every_n_epochs=1,
                        )
                    )
                    callbacks.append(MyLoggerCallback(dirpath=os.path.join(MODEL_PATH, file_dir)))

                trainer = pl.Trainer(
                    max_epochs=n_epochs,
                    accelerator="cpu",  # if str(device).startswith("cuda") else "cpu",
                    logger=loggers,
                    log_every_n_steps=log_every_n_steps,
                    enable_model_summary=True,
                    enable_progress_bar=True,
                    enable_checkpointing=CHECKPOINTING,
                    check_val_every_n_epoch=check_val_every_n_epoch,
                    devices=1,  # if str(device).startswith("cuda") else 1,
                    num_sanity_val_steps=0,
                    callbacks=callbacks,
                    gradient_clip_val=gradient_clip_val,
                    default_root_dir=str(MODEL_PATH),
                    gradient_clip_algorithm="value",
                    deterministic=False,  # "warn",
                )


                if use_latents and n_epochs_pretrain_latents > 0:

                    pretrain_callbacks = [
                        RichProgressBar(refresh_rate=1),
                        # GenerateCallback(
                        #     str(Path(final_plot_name).with_suffix("")) + "_pretrain",
                        #     plot_epoch_callback=plot_epoch_callback,
                        #     true_beta=beta.cpu().numpy(),
                        # ),
                    ]

                    if swa > 0:
                        pretrain_callbacks.append(StochasticWeightAveraging(0.01, swa_epoch_start=swa))

                    pretrain_callbacks.append(MyLoggerCallback(dirpath=os.path.join(MODEL_PATH, file_dir)))

                    pretrainer = pl.Trainer(
                        max_epochs=n_epochs_pretrain_latents,
                        accelerator="cpu",  # if str(device).startswith("cuda") else "cpu",
                        logger=loggers,
                        log_every_n_steps=log_every_n_steps,
                        enable_model_summary=True,
                        enable_progress_bar=True,
                        enable_checkpointing=CHECKPOINTING,
                        check_val_every_n_epoch=check_val_every_n_epoch,
                        devices=1,  # if str(device).startswith("cuda") else 1,
                        num_sanity_val_steps=0,
                        callbacks=pretrain_callbacks,
                        gradient_clip_val=gradient_clip_val,
                        default_root_dir=str(MODEL_PATH),
                        gradient_clip_algorithm="value",
                        deterministic=False,  # "warn",
                    )

                    print("PRETRAINING LATENTS!")
                    start_time = time.time()
                    model.train_only_likelihood = True
                    # assert False
                    pretrainer.fit(model, train_loader, validation_loader)
                    end_time = time.time()
                    model.train_only_likelihood = False

                # try:
                start_time = time.time()
                # assert False
                trainer.fit(model, train_loader, validation_loader)
                end_time = time.time()
                print(f"Training took {end_time - start_time:.2f} seconds")

                plot_training_results(
                    trainer,
                    model,
                    model.beta.detach().cpu().numpy(),
                    None,
                    scale_l1,
                    scale_kl,
                    scale_spectral,
                    scale_lyapunov,
                    final_plot_name,
                    callback=False,
                )

