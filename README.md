# Compress, structure, evolve: autoencoders, a VAE and a Koopman autoencoder on one flow dataset

These five notebooks follow one dataset through three ideas of reduced-order modelling with neural networks. The data are 1,000 snapshots of the velocity (u, v) of a 2-D simulation of viscous flow past two collinear plates at Reynolds number 100, on a 128 × 64 grid: 16,384 numbers per snapshot. Stage 1 compresses each snapshot into 10 numbers with a dense and a convolutional autoencoder and compares them with POD, the best linear compression of the same size. Stage 2 studies that latent space (how the error falls with its size, what its coordinates trace over time) and trains a variational autoencoder whose latent space can be sampled. Stage 3 tries to evolve the latent state in time with a Koopman autoencoder, a linear operator between an encoder and a decoder, against persistence and DMD. Stage 4 repeats the key comparisons on equal terms, with POD held to the same validation snapshots and a contiguous split in time for the rollouts. Stage 5 trains the Koopman model to forecast (a multi-step loss and a spectral penalty) and tests whether the autoencoder's advantage over POD survives a split in time. Every training is repeated with three seeds.

Portfolio page: https://vamsigudipati.github.io/latent-dynamics-ae-vae-koopman.html

## Notebooks

| Notebook | Stage | Results | Models | Animations |
| --- | --- | --- | --- | --- |
| `notebooks/01_autoencoders_and_pod.ipynb` | Stage 1 — Compressing a flow field: dense and convolutional autoencoders against POD | `results/01_autoencoders_and_pod/exercises/` | `models/01_autoencoders_and_pod/` (release `models_stages_1_3_4.tar`) | `animations/01_*` |
| `notebooks/02_latent_space_and_vae.ipynb` | Stage 2 — Structuring the latent space: latent size, trajectories and a variational autoencoder | `results/02_latent_space_and_vae/exercises/` | `models/02_latent_space_and_vae/` (release `models_stage2_latent_space_and_vae.tar`) | `animations/02_*` |
| `notebooks/03_koopman_autoencoder.ipynb` | Stage 3 — Evolving the latent state: a Koopman autoencoder against persistence and DMD | `results/03_koopman_autoencoder/exercises/` | `models/03_koopman_autoencoder/` (release `models_stages_1_3_4.tar`) | `animations/03_*` |
| `notebooks/04_rollout_and_baselines.ipynb` | Stage 4 — Latent size and rollouts: what the networks buy over POD and DMD | `results/04_rollout_and_baselines/` | `models/04_rollout_and_baselines/` (release `models_stages_1_3_4.tar`) | — |
| `notebooks/05_koopman_trained_to_forecast.ipynb` | Stage 5 — A Koopman model trained to forecast: multi-step loss, a spectral penalty, and compression on a split in time | `results/05_koopman_trained_to_forecast/` | `models/05_koopman_trained_to_forecast/` (seed-0 operators as `.npy` in git; the `.keras` encoders, decoders and the time-split autoencoder are not in git and not in a release: each is retrained in 1–4 minutes when the notebook runs) | — |

Each notebook is also exported as HTML next to it, with all outputs. The `exercises/` folders hold the tables (CSV) and figures (PNG) of the experiments and of the three-seed repeats (`seeds_runs.csv`, `seeds_summary.csv`); Stages 4 and 5 write their tables and figures directly into their results folders. `models/` keeps the small metadata files in git; the trained networks (`.keras`, up to 200 MB each) are release files. `animations/` holds the training videos and their final frames.

The outputs in the notebooks come from my single run of each notebook. For this repository I changed only text afterwards: a header cell and a paths cell in each notebook, the path lines (data, models, outputs), comments, labels and printed text. The notebooks were not executed again in this layout; the paths cell is therefore the one cell without an output.

## Data and release files

The dataset is too large for git (124 MB) and comes with the repository's first release, together with the trained models:

| Release file | Size | Goes to | SHA-256 |
| --- | --- | --- | --- |
| `G1Re100_coarse.npz` (arrays `x`, `y`, `UV` of shape (1000, 128, 64, 2)) | 124 MB | `release_assets/G1Re100_coarse.npz` | `6d9b6c4f445c6a76d18d54486b7bc47fe621eabb9b9090d33dc25fb61252dde3` |
| `models_stages_1_3_4.tar` (optional): Stage 1 dense and convolutional autoencoders with their encoders and decoders, Stage 3 Koopman autoencoder with encoder, operator and decoder, Stage 4 seed-0 Koopman autoencoder | 527 MB | extract at the repository root, giving `models/01_…/`, `models/03_…/`, `models/04_…/` | `6f72ab93e69f91e421664b6592f948a23ca3951fbf1dc2272f2b40c744b48ee3` |
| `models_stage2_latent_space_and_vae.tar` (optional): the five autoencoders of the latent-size sweep, the "reference" autoencoder and the VAE encoder and decoder | 1.68 GB | extract at the repository root, giving `models/02_latent_space_and_vae/` | `7e43a4151ab770364adc3f29fd15cabd4a3447bda80324c88423d2e7972757a9` |

`release_assets/` is ignored by git, and so are the `.keras` files under `models/`. Every stage trains its own models when it runs, so the model archives are optional: they hold the networks of my run, for inspection without retraining. Every model in them was trained by my runs; no pretrained model is used. If the dataset file is missing, the data cell of Stages 1–3 downloads it from the address in the notebook into `release_assets/`.

## Setting up and running

I ran everything on a laptop (Apple M3 Max, 48 GB) with Python 3.12, TensorFlow 2.18.1, Keras 3.6 and the `tensorflow-metal` plug-in for the Apple GPU. The frozen environment is in `requirements.txt`; the videos use the ffmpeg binary that `imageio-ffmpeg` installs.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name py312-tf-metal-studio
mv G1Re100_coarse.npz release_assets/
cd notebooks
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 01_autoencoders_and_pod.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 02_latent_space_and_vae.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 03_koopman_autoencoder.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 04_rollout_and_baselines.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 05_koopman_trained_to_forecast.ipynb
```

All paths inside the notebooks are relative to `notebooks/`, so run them from there. On the laptop, Stage 1 took 66 minutes (CPU), Stage 2 38 minutes, Stage 3 14 minutes, Stage 4 21 minutes and Stage 5 26 minutes, one at a time. Stage 5 reads Stage 4's device timings and tables and Stage 1's three-seed summary, so run those first.

**Why these devices.** Before training, I timed one training step of each network inside `model.fit` (batch 32, median of 30 steps after 10 warm-up steps) on the CPU and on the Apple GPU (`results/04_rollout_and_baselines/device_timing_*.csv`). In milliseconds per step: dense autoencoder 19.5 vs 9.6, dense VAE 19.7 vs 11.3, Koopman autoencoder 10.3 vs 9.9, so these train on the GPU (Stage 5's models use the same dense layers and reuse these timings; the environment is unchanged); but the convolutional autoencoder 37.7 vs 624, the two-block one 63.5 vs 729 and the convolutional VAE 37.8 vs 625, so those train on the CPU, and the non-linear operator of Stage 3's Experiment 1 also on the CPU (10.4 vs 12.1). Stage 1 runs entirely on the CPU: with the GPU visible, the convolutional autoencoder pinned to the CPU still ran at about 6 s per epoch, six times its CPU-only rate, which would have taken it past my 45-minute limit per training; on the CPU alone it trains in 10 minutes. With the same seed and settings, a GPU training repeated to every printed digit in my runs.

**Why three seeds.** A single training gives one draw of the initial weights and of the batch order. Every training here takes under 15 minutes, so each is repeated with the seeds 0, 1 and 2 (seed 0 is the base run), and I report the mean, the standard deviation and the range. A difference smaller than that spread is not an effect. Two examples from these runs: the β = 1 VAE of Stage 2's Experiment 3 scored 0.449 in an earlier run and 0.641 ± 0.017 over three seeds here; and whether Stage 3's learned operator is stable changes with the seed.

## What the stages show

- Stage 1: at 10 latent numbers, the convolutional autoencoder (0.437 ± 0.006, mean relative error on the validation snapshots) beats the dense one (0.465 ± 0.009) in every seed, and both are far below POD of the same rank (0.757 when POD is fitted on the training snapshots only, as Stage 4 measures). The latent "modes" of an autoencoder are sensitivities, not a basis that adds up to the field.
- Stage 2: the advantage over POD holds at every latent size from 5 to 50; the VAE reconstructs better than the plain autoencoder at 10 numbers (0.434 ± 0.010 against 0.497 ± 0.017 in Stage 2's sweep) and can be sampled, but fields decoded from prior draws carry only a few per cent of the real fluctuation energy.
- Stage 3: the Koopman autoencoder's one-step prediction (0.531 ± 0.017) is worse than persistence (0.366) in every seed, though better than DMD on 10 POD coefficients (0.721).
- Stage 4: on equal terms the compression result stands (d = 10: POD 0.757, autoencoder 0.499, VAE 0.448), but the forecast does not. On held-out starts after a contiguous split, the Koopman autoencoder is worse than predicting the mean flow from the first step and, averaged over three seeds, worse than DMD at every horizon up to 50 steps (1.42 against 1.10 at 50), and in two of three seeds its operator has eigenvalues outside the unit circle.
- Stage 5, training to forecast: none of the three Koopman models (one-step loss; multi-step loss over 1–5 steps; multi-step loss plus a spectral penalty) beats DMD at any of the 50 horizons. The multi-step loss removes the one-step model's blow-up (error at 50 steps 1.42 with a seed range of 0.62 → 1.20 with a range of 0.10) but the model settles near the mean flow (error 1), and the spectral penalty pins every eigenvalue on the unit circle to four decimals in every seed without improving the forecast.
- Stage 5, compression on a split in time: fitted on the first 800 snapshots and scored on the last 200, the dense autoencoder (0.96, range 0.95–0.97 over three seeds) is worse than POD with a training basis (0.79). The advantage over POD seen in Stages 1, 2 and 4 (0.46–0.50 against 0.76) belongs to the random split, where every scored snapshot has neighbours in time in the training set.
