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
DS/cifar10/
  train/
    airplane/
      0001.png
    automobile/
      0001.png
    ...
  test/
    airplane/
      0001.png
    automobile/
      0001.png
    ...
```

Each image lives directly in its class directory. The directory name is its
label; matching class names across `train/` and `test/` get the same integer
label. Supported formats are PNG, JPEG, BMP, WebP, and TIFF. Class directories
and filenames are sorted for reproducible exports. Set `dataset_path` in
`cfgs/cfg.py` or pass `--dataset-path` to use another dataset root.

All images in `train/` are used for training, and the labeled images in `test/`
supply evaluation metrics. Test classes must also exist in `train/`. Console
summaries call these metrics `test`. Because evaluation loss controls the
scheduler and best checkpoint, reserve a separate untouched set if you need an
unbiased final test score.

Evaluation runs every epoch and ReduceLROnPlateau monitors `eval_loss`. The
progress bar and model summary are disabled. Only train/evaluation loss and
accuracy are printed every five epochs; warnings and errors remain visible.
All epoch metrics, including learning rate, are saved even between print epochs.
Training accuracy is measured on augmented training batches. A singleton final
training batch is dropped for BatchNorm; evaluation and export retain every row.

## Experiment artifacts

Each run creates `log_dir/YYMMDD-hh_mm_ss/` containing:

```text
config.json            # configuration, image counts, versions
class_to_idx.json      # stable alphabetical label -> integer mapping
hparams.yaml
metrics.csv            # train/evaluation metrics and learning rate
checkpoints/           # best evaluation-loss checkpoint and last.ckpt
out/
  embeddings_train.pt  # float32 tensor [N_train, 512] for ResNet-18
  labels_train.pt      # int64 tensor [N_train]
  image_paths_train.csv
  embeddings_test.pt   # float32 tensor [N_test, 512] for ResNet-18
  labels_test.pt       # int64 tensor [N_test]
  image_paths_test.csv
```

Each split's three files contain its images in sorted class/filename order.
Row `i` of `embeddings_<split>.pt`, element `i` of `labels_<split>.pt`, and
data row `i` of `image_paths_<split>.csv` refer to the same image. The CSVs
have an `image_path` header and paths relative to the project root.
Embeddings come from the final epoch's pooled features immediately before the
classifier, with augmentation disabled. `class_to_idx.json` decodes labels.
Set `sae_smples` in `cfgs/cfg.py` (or pass `--sae-smples`) to a percentage from
greater than 0 through 100 (for example, `25.0` exports about a quarter of each
split). Each split's count is rounded to the nearest image, with a minimum of
one. The selection is seeded, balanced across classes as far as each class
allows, and Gaussian-weighted toward the middle of each class's
sorted filenames. Selected rows retain sorted class/filename order. The default
`None` exports every image; classifier training and evaluation always use their
full splits.

```python
import csv
import torch

run = "log_dir/YYMMDD-hh_mm_ss"
for split in ("train", "test"):
    embeddings = torch.load(f"{run}/out/embeddings_{split}.pt", weights_only=True)
    labels = torch.load(f"{run}/out/labels_{split}.pt", weights_only=True)
    with open(f"{run}/out/image_paths_{split}.csv", newline="") as f:
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

The SAE reads `out/embeddings_train.pt` directly, infers its input dimension,
and uses raw training embeddings without additional normalization. Test
embeddings remain available for separate analysis. The default hidden dimension
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
  --images-dir DS/cifar10/test \
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
