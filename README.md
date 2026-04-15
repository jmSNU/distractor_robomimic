# 🤖 Robomimic Dataset Augmentation via Camera and Lighting Perturbation

This repository provides a pipeline for augmenting Robomimic demonstration datasets by performing state-consistent rollouts with controlled visual perturbations including:

- camera pose jitter
- lighting variation
- offscreen rendering perturbation

The goal is to improve robustness of visuomotor policies under domain shift.

---

# Key Idea

We replay original demonstrations using ground-truth simulator states and generate new trajectories with randomized visual conditions:

- same actions
- same physical trajectory
- different visual observations

This decouples dynamics from perception.

---

# Pipeline Overview

Original HDF5 Demo → State Replay in Robosuite → Camera and Lighting Perturbation per step → Re-render Observations → Augmented HDF5 Dataset

---

# Installation

pip install -r requirements.txt

---

# Dataset Format

Input format

data/
  demo_0/
    states
    actions
    obs/
      agentview_image
      robot0_eye_in_hand_image

Output format

data/
  demo_0/
    obs/
    next_obs/
    actions
    rewards
    dones

---

# Augmentation Strategy

## Camera perturbation
- position noise
- rotation noise (axis-angle)

## Lighting perturbation
- diffuse color noise
- light position jitter


# Generate Visualization



# Augmentation Script

python augment_demo.py \
  --input data/lift_ph.hdf5 \
  --output data/lift_aug_ph.hdf5

---


# Notes

- Dataset must include states
- Camera perturbation applied per step
- Lighting perturbation may accumulate if not reset
- Use render_offscreen=True for headless execution

---
