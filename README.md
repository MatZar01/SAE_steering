# SAE_steering

## Run

Dependencies are listed in `requirements.txt`

```bash
conda activate saes
python main.py
# Optional overrides:
python main.py --epochs 50 --batch-size 32 --learning-rate 0.0003
```

Edit `cfgs/cfg.py` for dataset paths, model name, optimizer, plateau scheduler,
augmentations, workers, precision, and other experiment parameters. Relative paths
are resolved from the project root, independent of the working directory.
`src/models.py` has a model registry; add a constructor exposing `forward_features`
and `embedding_dim` to make another model selectable by its string name.

The default model is ImageNet-pretrained ResNet-18, fine-tuned end to end with
AdamW. Its weights are downloaded by torchvision on first use. `--no-pretrained`
supports offline smoke runs. Training uses random crops, horizontal flips,
rotation, color jitter, Gaussian blur, and random masking. Evaluation and export
use deterministic resize/center crop and ImageNet normalization.

## Dataset and evaluation

The default layout is:

```text
DS/cifar-10/
  train/1.png ...
  test/1.png ...
  trainLabels.csv       # id,label
```

Only training labels are supplied in this dataset. By default, a reproducible,
stratified 10% holdout of the labeled images supplies validation metrics; the
unlabeled `test/` images cannot supply loss or accuracy. Console summaries call
these metrics `val`. To use labeled test images, set `test_labels_csv` in the
config, or pass `--test-labels-csv testLabels.csv`, with the same `id,label`
schema. In that mode, all training rows are used for training and console
summaries call evaluation metrics `test`. Because evaluation loss controls the
scheduler and best checkpoint, this evaluation set acts as validation; reserve
a separate untouched set if you need an unbiased final test score.

Evaluation runs every epoch and ReduceLROnPlateau monitors `eval_loss`. The
progress bar and model summary are disabled. Only train/evaluation loss and
accuracy are printed every five epochs; warnings and errors remain visible.
All epoch metrics, including learning rate, are saved even between print epochs.
Training accuracy is measured on augmented training batches. A singleton final
training batch is dropped for BatchNorm; evaluation and export retain every row.

## Experiment artifacts

Each run creates `log_dir/YYMMDD-hh_mm_ss/` containing:

```text
config.json            # configuration, evaluation source, counts, versions
class_to_idx.json      # stable alphabetical label -> integer mapping
split.csv              # exact train/evaluation membership
hparams.yaml
metrics.csv            # train/evaluation metrics and learning rate
checkpoints/           # best evaluation-loss checkpoint and last.ckpt
out/
  embeddings.pt        # float32 tensor [N, 512] for ResNet-18
  labels.pt            # int64 tensor [N]
  image_paths.csv      # image_path header, then N paths relative to project root
```

All three exports follow the original `trainLabels.csv` row order, including
any held-out training images. Row `i` of `embeddings.pt`, element `i` of
`labels.pt`, and data row `i` of `image_paths.csv` refer to the same image.
Embeddings come from the final epoch's pooled features immediately before the
classifier, with augmentation disabled. `class_to_idx.json` decodes labels and
`split.csv` identifies which exported images were held out.

```python
import csv
import torch

run = "log_dir/YYMMDD-hh_mm_ss"
embeddings = torch.load(f"{run}/out/embeddings.pt", weights_only=True)
labels = torch.load(f"{run}/out/labels.pt", weights_only=True)
with open(f"{run}/out/image_paths.csv", newline="") as f:
    paths = [row["image_path"] for row in csv.DictReader(f)]
assert len(embeddings) == len(labels) == len(paths)
```

## Sparse autoencoders

SAE training uses Overcomplete 0.2.2 models with PyTorch Lightning. Set
`model_log_dir` in `cfgs/cfg_sae.py` to a completed image experiment, or pass it
on the command line:

```bash
conda activate saes
python main_sae.py --model-log-dir log_dir/YYMMDD-hh_mm_ss --model-name basic
python main_sae.py --model-log-dir log_dir/YYMMDD-hh_mm_ss --model-name topk --top-k 32
python main_sae.py --model-log-dir log_dir/YYMMDD-hh_mm_ss --model-name batch_topk --top-k 32
```

`src/models_sae.py` provides three choices:

| Model name | Sparsity | Training objective |
| --- | --- | --- |
| `basic` | ReLU codes with an L1 penalty | MSE + `l1_coefficient * mean(sum(abs(codes)))` |
| `topk` | At most `top_k` nonzero codes per image | MSE |
| `batch_topk` | Global threshold selecting `batch_size * top_k` activations during training | MSE |

All variants use Overcomplete's linear/ReLU encoder without BatchNorm and its
L2-normalized dictionary. Batch Top-K scales its budget for the actual batch
size, including a short final batch. Its upstream threshold rule can retain
extra activations when values tie. Evaluation uses the learned running threshold,
which is saved with the model; inference is independent of the other images in
the batch, and its active-feature count is not constrained to exactly `top_k`.
These adaptations follow the [Overcomplete BatchTopK implementation](https://github.com/KempnerInstitute/overcomplete/blob/main/overcomplete/sae/batchtopk_sae.py).
A small compatibility alias supports Overcomplete's plotting imports with recent
Matplotlib versions; no environment downgrade is needed.

The SAE reads `out/embeddings.pt` directly, infers its input dimension, and uses
raw embeddings without additional normalization. The default hidden dimension
is 2048. A seeded 10% embedding holdout is used for validation and the plateau
scheduler; its row indices are saved. The image labels are not used. The progress
bar stays disabled and reconstruction loss, MSE, and mean active-feature count
(`l0`) print every five epochs. Accuracy is not defined for this reconstruction
task. All epoch metrics are also saved to CSV.

Every SAE experiment gets a separate directory inside its source experiment:

```text
log_dir/<image-run>/SAE/<YYMMDD-hh_mm_ss>/
  config.json         # SAE settings, source path, dimensions, package versions
  hparams.yaml
  metrics.csv         # loss, MSE, L1, L0 and learning rate
  split_indices.pt    # train/val indices into the source embeddings tensor
  model.pt            # final SAE architecture, weights and inference threshold
  last.ckpt           # final Lightning checkpoint, including optimizer state
```

Print hidden SAE embeddings for images in a directory (searched recursively):

```bash
python test_sae.py \
  --log-dir log_dir/<image-run>/SAE/<sae-run> \
  --images-dir DS/cifar-10/test \
  --device cpu
```

`--log-dir` also accepts the source image experiment or its `SAE` directory;
these select the latest completed SAE run. Use the exact SAE run path to select
a particular model. The script loads the original image model's
`checkpoints/last.ckpt`, recreates its deterministic preprocessing from
`config.json`, computes image features, and prints one JSON record per image
with its path and full SAE hidden code (after ReLU/sparsification). It does not
download ImageNet weights. Keep the source image checkpoint/config with the SAE;
`--model-log-dir` can override their location after moving files.

Run the integration checks with `python -m unittest discover -s tests -v`.
