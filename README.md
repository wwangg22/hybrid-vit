# Hybrid Vision Transformers

Mix and match Performer and regular attention layers in Vision Transformers.

## Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running Experiments

We've got two main scripts to run experiments:

### Run All Experiments
```bash
bash src/run_all_experiments.sh
```

This runs the full suite of experiments across different model configurations.

### Run Long Experiments
```bash
bash src/run_long_exp.sh
```

This runs longer, more thorough experiments (takes more time).

## What's Inside

- **VanillaViT**: Standard ViT with all regular attention (baseline)
- **AlternatingViT**: Switches between Performer and regular attention
- **PerformerFirstViT**: Performer layers first, then regular
- **RegularFirstViT**: Regular layers first, then Performer
- **CustomPatternViT**: Whatever pattern you want

All models work on MNIST and CIFAR-10.

## Training a Single Model

If you just want to train one model:

```bash
python src/train.py --config configs/your_config.yml
```

## Results

Results and model checkpoints will be saved to the output directory specified in your config file. Training metrics are logged to Weights & Biases.