# SITCON2021-DL-Adversarial-Attack-Tutorial
An simple tutorial for generating adversarial examples against any DL model

## Quick start
### Step 1: Download imagenette2 dataset
To simplify the experiment, we only use a subset of ImageNet here. The imagenette2 dataset contains ten categories from the original ImageNet.
```
./getData.sh
```
### Step 2: Fine-tune the pretrained model
Please fine-tune the model to fit the subset with ten categories.
```
python3 train.py
```
### Step 3: Generate and visualize adversarial examples
Please open the `Generate.ipynb` and run it to generate adversarial examples
