# CT Reconstruction with Flow Matching based model

This repository implements and extends **flow-matching-based** models for **Computed Tomography (CT) reconstruction**.  
It adapts the [`flow_matching`](https://github.com/facebookresearch/flow_matching) package and integrates it with a finite-angle CT Radon operator for inverse problem solving.

---

## 1. Data Preparation

### Dataset
- **Source:** [AAPM CT Dataset](https://github.com/taehoon-yoon/CT-denoising-using-CycleGAN?tab=readme-ov-file)  
- **Direct Download:** [Google Drive Link](https://drive.google.com/file/d/1Ov6yyzbnCC_gYNuk6RS6EfvVAoSqKGUC/view?usp=sharing)

### Downsampling & Normalization

> **Modifications:**  
> - Applied additional downsampling and normalization to scale pixel values to **[0, 1]**.  
> - Normalization ensures stable reconstruction — without it, reconstructed images appeared distorted.  
> - During training, all input images are in [0,1].  
> - In latent variable optimization, the predicted sinogram from `physics(x_hat)` is compared with the normalized true sinogram.  
> - For visualization, `x_hat` is remapped back to [0,1].

**Processed dataset:** [Google Drive Link](https://drive.google.com/file/d/1OB-VirFBX22-zQX2QMpVvmwYeE3WUVD1/view?usp=drive_link)

---

## 2. Dataset Loader

- Implemented `ct_dataset.py` for loading preprocessed CT images.

---

## 3. Training Flow Matching Model

### Script
`train_flow_ct.py` — adapted from `train_flow_matching.py` in **NeuralLatentInversion** (send by Alex).

**Model Config (`model_cfg`):**
```python
model_cfg = {
    "in_channels": 1,
    "model_channels": 32,
    "out_channels": 1,
    "num_res_blocks": 2,
    "attention_resolutions": [8],
    "dropout": 0.0,
    "channel_mult": [1, 2, 2, 4],
    "conv_resample": False,
    "dims": 2,
    "num_classes": None,
    "use_checkpoint": True,
    "num_heads": 1,
    "num_head_channels": -1,
    "num_heads_upsample": -1,
    "use_scale_shift_norm": True,
    "resblock_updown": False,
    "use_new_attention_order": True,
    "with_fourier_features": False,
    "max_period": 2.0
}
````

### Sampling
* `sample_flow_ct.py` for unconditional sampling.

### Unconditional Sampling Visualization

`visualization.py`

* Visualizes flow-based model generating images from pure noise.
* `sampler_steps=50`, intermediate images saved in `viz/`.

---

## 4. Latent Variable Optimization

Adapted `flow_latent_opt.py` and `flow_latent_opt_adjoint.py` from NeuralLatentInversion:

* Replaced MNIST dataset with CT dataset.
* Replaced forward operator with finite-angle CT Radon transform:
  [DeepInv Tomography](https://deepinv.github.io/deepinv/api/stubs/deepinv.physics.Tomography.html#deepinv.physics.Tomography)

### Methods

* `flow_latent_opt_ct.py` — midpoint solver
* `flow_latent_opt_adjoint_ct.py` — adjoint solver

---

## 5. Experiments

Script: `4methods.py`
**Settings:**

* Tomography: 90 angles, 0.05 noise
* Compare Unrolled vs Adjoint methods
* Initialization: Random vs D-Flow
* 6 ODE time steps, L-BFGS optimizer

Run example saved in 4methods.sh:

```bash
python 4methods.py \
    --method unrolled \
    --init dflow \
    --seeds 0,142,123,3407 \
    --iter_max 200 \
    --device cuda \
    --output_dir 4methodsresults/unrolled_dflow
```

---

## 6. Extended Experiments

### Sparse View Reconstruction

`sparse30/` and `sparse60/` (30 or 60 projection angles)
Script: `sparse.py`

### Limited Angle Reconstruction

`limited90/` and `limited60/` — fixed angles
`limited90new/` and `limited60new/` — dynamic angles
Script: `limited.py`

### Latent Norm Monitoring

`monitor_norm.py` — records ‖z‖ every N iterations and plots curve.

### ODE Solver Variations

* `solver_choice.py` — different forward solvers (`midpoint`, `rk4`, `euler`)
* `adjoint_solver.py` — different reverse solvers

---

## 7. References

* Flow Matching package: [https://github.com/facebookresearch/flow\_matching](https://github.com/facebookresearch/flow_matching)
* CT Dataset:

  * [https://github.com/taehoon-yoon/CT-denoising-using-CycleGAN](https://github.com/taehoon-yoon/CT-denoising-using-CycleGAN)
  * [Download Link](https://drive.google.com/file/d/1Ov6yyzbnCC_gYNuk6RS6EfvVAoSqKGUC/view?usp=sharing)
* Radon Operator: [DeepInv Tomography](https://deepinv.github.io/deepinv/api/stubs/deepinv.physics.Tomography.html#deepinv.physics.Tomography)

---

## 8. Folder Structure

```
.
├── ct_dataset.py
├── train_flow_ct.py
├── train_flow_matching.py
├── sample_flow_ct.py
├── flow_mathcing_ct.py
├── flow_mathcing_ct.yaml
├── visualization.py
├── sample_flow_ct.py
├── sample_epoch_0.png
├── 4methods_5imageset.py
├── 4methods_5imageset.sh
├── 4methods.py
├── 4methods.sh
├── sparse.py
├── sparse.sh
├── limited.py
├── limited.sh
├── monitor_norm.py
├── monitor_norm.sh
├── solver_choice.py
├── solver_choice.sh
├── adjoint_solver.py
├── adjoint_solver.sh
├── flow_mathcing/
├── models/
├── 4methods_results/
├── 4methods_resultslog/
├── 4methods_5imageset/
├── 4methods_5imagesetlog/
├── limited60new/
├── limited60newlog/
├── limited90new/
├── limited90newlog/
├── sparse30/
├── sparse30log/
├── sparse60/
├── sparse60log/
├── monitor_norm/
├── monitor_normlog/
├── adjoint_solvereuler/
├── adjoint_solvereulerlog/
├── adjoint_solverrk4/
├── adjoint_solverrk4log/
├── solver_choice_euler/
├── solver_choice_eulerlog/
├── solver_choice_rk4/
├── solver_choice_rk4log/
├── viz/
└── 
```
