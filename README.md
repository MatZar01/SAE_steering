# SAE_steering

## Run

Dependencies are listed in `requirements.txt`

```bash
conda activate saes
python main.py
# Optional overrides:
python main.py --epochs 50 --batch-size 32 --learning-rate 0.0003
python main.py --model-name blip --batch-size 16
python main.py --model-name dino --batch-size 32
python main.py --model-name blip --export-batch-size 2
python main.py --model-name dino --sae-model-name topk
```

To run image training, SAE training, neighborhood steering, and active-image
inference in one sequence, edit the four dictionaries in `run_pipeline.py` and run:

```bash
python run_pipeline.py
```

For example, the override dictionary can contain:

```python
PIPELINE_OVERRIDES = {
    "image": {"model_name": "dino", "epochs": 20,
              "augments": {"image_size": 224}, "scheduler": {"patience": 2}},
    "sae": {"model_name": "topk", "concept_expansion_factor": 4,
            "top_k": 32, "epochs": 30},
    "steering": {"max_anchors": 128, "max_neighborhood_size": 16},
    "inference": {"top_active_images": 3},
}
```

Every key corresponds to a field in `cfgs/cfg.py` or `cfgs/cfg_sae.py`;
unspecified fields keep their defaults. The image stage chooses the base model.
The SAE stage's `model_name` chooses its autoencoder variant. The pipeline passes
the exact image run to SAE training, both run paths to steering, and the exact
SAE and steering runs to inference. Their linked run paths are set automatically;
leave them out of the overrides. Results remain nested under the chosen image
`log_dir`.

Edit `cfgs/cfg.py` for dataset paths, model name, optimizer, plateau scheduler,
augmentations, workers, precision, and other experiment parameters. Relative paths
are resolved from the project root, independent of the working directory.
`src/models.py` has a model registry; each constructor exposes `forward_features`
and `embedding_dim` so its features can be exported for SAE training. The image
model is selected in `cfgs/cfg.py` or with `--model-name` in `main.py`. The linked
`cfgs/cfg_sae.py` controls the separate SAE stage and its `model_name` chooses
`basic`, `topk`, or `batch_topk`. An image run's folder name includes both model
names. In the pipeline, the SAE model name comes from the `sae` overrides;
standalone `main.py` uses the SAE config default or `--sae-model-name`.

ResNet-18 can be fine-tuned end to end with AdamW. BLIP and DINO are fixed
encoders with a new linear classifier;
their train and test features are extracted once, then only the classifier is
optimized from the saved features. Extraction displays a progress bar for each
split. The number of outputs is taken from the training class folders. Their
pretrained weights are downloaded on first use. For pretrained BLIP/DINO, the
full feature cache lives under `features/<model>/<cache-key>/` at the project
root. Later runs with the same images, encoder and preprocessing reuse it without
loading the pretrained encoder. Changes to image files or image size create a
new cache key; classifier settings and `sae_smples` can change while reusing it.
`--no-pretrained` skips the download for offline smoke runs and gives a randomly
initialized, frozen encoder; those features remain local to that run.

| Model name | Checkpoint | Frozen features | Suggested input size |
| --- | --- | --- | --- |
| `resnet18` | torchvision ImageNet-1k weights | No; the encoder is fine-tuned | 224 |
| `blip` | [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b) | 1408-dimensional vision CLS embedding | 224 |
| `dino` | [facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) | 768-dimensional vision CLS embedding | 224 |

Only BLIP-2's vision encoder is instantiated; its Q-Former and language model
are unnecessary for this classifier. DINOv3's pretrained representation has no
task classifier. Each gets a new classifier sized to the dataset; exported
embeddings are taken before it. DINOv3's checkpoint requires accepting its
model license on Hugging Face and signing in before its weights can be downloaded.
Both checkpoints use 224-pixel inputs by default. BLIP-2 supports positional
embedding interpolation at other sizes; DINOv3 accepts sizes divisible by 16.
BLIP uses the checkpoint's normalization, while ResNet and DINO use ImageNet
normalization. ResNet training uses random crops, horizontal flips, rotation,
color jitter, Gaussian blur, and random masking. Frozen encoders use deterministic
preprocessing for both feature extraction and classifier training. Evaluation and
export use a square resize for BLIP and DINO, or resize/center crop for ResNet, followed by the
selected model's normalization.

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
training progress bar and model summary are disabled. Train/evaluation loss and
accuracy are printed every five epochs; ResNet export reports progress, and
BLIP/DINO feature extraction has a per-split progress bar. Warnings and errors
remain visible.
All epoch metrics, including learning rate, are saved even between print epochs.
ResNet training accuracy is measured on augmented training batches. A singleton
final ResNet training batch is dropped for BatchNorm; evaluation and export
retain every row.
Embedding export uses `export_batch_size` (default 4) independently of the
training `batch_size`, to keep vision encoder export within memory limits. Set
it in `cfgs/cfg.py`, in the pipeline image overrides, or with
`--export-batch-size` in `main.py`.

## Experiment artifacts

Each image run creates `log_dir/YYMMDD-hh_mm_ss_<vision>_<sae>/` containing:

```text
config.json            # configuration, image counts, versions
class_to_idx.json      # stable alphabetical label -> integer mapping
hparams.yaml
metrics.csv            # train/evaluation metrics and learning rate
checkpoints/           # best evaluation-loss checkpoint and last.ckpt
frozen_encoder.pt       # BLIP/DINO only: saved fixed encoder weights
out/
  embeddings_train.pt  # float32 tensor [N_train, embedding_dim]
  labels_train.pt      # int64 tensor [N_train]
  image_paths_train.csv
  embeddings_test.pt   # float32 tensor [N_test, embedding_dim]
  labels_test.pt       # int64 tensor [N_test]
  image_paths_test.csv
```

Pretrained frozen runs also use a shared cache outside `log_dir`:

```text
features/<model>/<cache-key>/
  manifest.json         # cache identity; written only after extraction completes
  frozen_encoder.pt     # reused by image runs
  features_train.pt, feature_labels_train.pt, feature_paths_train.csv
  features_test.pt, feature_labels_test.pt, feature_paths_test.csv
```

The run's `config.json` records `feature_cache_dir`. Unsampled SAE exports and
the encoder file link to this cache, avoiding duplicate storage.
Offline `--no-pretrained` runs keep the full feature files in their own `out/`.

Each split's three files contain its images in sorted class/filename order.
Row `i` of `embeddings_<split>.pt`, element `i` of `labels_<split>.pt`, and
data row `i` of `image_paths_<split>.csv` refer to the same image. The CSVs
have an `image_path` header and paths relative to the project root.
ResNet embeddings come from the final epoch's pooled features immediately before
the classifier. BLIP/DINO embeddings come from the frozen encoder before
classifier training. In both cases export has augmentation disabled.
`class_to_idx.json` decodes labels. BLIP/DINO classifier checkpoints contain
only the linear head; `frozen_encoder.pt` stores the encoder once.
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

run = "log_dir/YYMMDD-hh_mm_ss_dino_topk"
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
python main_sae.py --model-log-dir log_dir/<image-run> --model-name basic
python main_sae.py --model-log-dir log_dir/<image-run> --model-name topk --top-k 32
python main_sae.py --model-log-dir log_dir/<image-run> --model-name batch_topk --top-k 32
python main_sae.py --model-log-dir log_dir/<image-run> --concept-expansion-factor 2
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
is four times the input feature dimension. Set `concept_expansion_factor` to a
positive integer; for example, a factor of 2 with 512-dimensional image features
creates 1024 SAE concepts. The resolved count is saved as `nb_concepts` in the
model architecture for checkpoint loading. A seeded 10% embedding holdout is
used for validation and the plateau scheduler; its row indices are saved. The
image labels are not used. The progress bar stays disabled and reconstruction
loss, MSE, mean active-feature count (`l0`), R², and dead-neuron count print every
five epochs for both training and validation. R² uses total squared reconstruction
error against the per-feature mean of that epoch's samples. A dead neuron never
has a positive code on that split during the epoch. Accuracy is not defined for
this reconstruction task. All epoch metrics are also saved to CSV.

Every SAE experiment gets a separate directory inside its source experiment:

```text
log_dir/<image-run>/SAE/<YYMMDD-hh_mm_ss>/
  config.json         # SAE settings, source path, dimensions, package versions
  hparams.yaml
  metrics.csv         # loss, MSE, L1, L0, R², dead neurons, learning rate
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
`checkpoints/last.ckpt` (and `frozen_encoder.pt` for BLIP/DINO), recreates its
deterministic preprocessing from
`config.json`, computes image features, and prints one JSON record per image
with its path and full SAE hidden code (after ReLU/sparsification). It does not
download ImageNet weights. Keep the source image checkpoint/config with the SAE;
`--model-log-dir` can override their location after moving files.

## Neighborhood-based SAE steering evaluation

`main_steering.py` implements point 1 of `SAE_tversky_group_steering.pdf` using
the saved image embeddings, labels, and SAE checkpoint. Set `sae_log_dir` in
`cfgs/cfg_sae.py` or pass an exact SAE run, its `SAE` directory, or the source
image run:

```bash
python main_steering.py --sae-log-dir log_dir/<image-run>/SAE/<sae-run>
# Evaluate every eligible anchor and neighborhood sizes 1 through 16:
python main_steering.py --sae-log-dir log_dir/<image-run>/SAE/<sae-run> \
  --all-anchors --max-neighborhood-size 16
```

The runner fits one linear classifier to the exported original training
representations and then freezes it. For each SAE latent, training support is
`z_k(x) > mean_train(z_k)`. Latents with no positive training activation or
empty support are excluded. Neighbors are ranked by symmetric Tversky overlap
of support sets, `|A∩B| / (|A∩B| + p(|A∖B| + |B∖A|))`, with `p=0.5` by default;
equal scores use latent index order. The default samples up to 128 eligible
anchors with seed 42 and evaluates every neighborhood size from 1 through 8.

Ablation zeros the neighborhood's codes, decodes the result, and passes it to
the fixed classifier. Accuracy drops are measured against the unmodified SAE
reconstruction on the exported test set; accuracy on original test embeddings
is saved as a reference. The class with the largest drop at size 1 is fixed as
that anchor's target at every larger size. Non-target damage is the unweighted
mean accuracy drop across the other observed classes. The support of a
neighborhood is the union of its latent supports. Its variance metrics use the
original training embeddings in that union, with population variance (`N` in
the denominator). Total variance and mean squared distance to the centroid
are equal by definition. The steering log also reports `active_state_variance`
and `active_state_cosine_similarity` for each group: these use the group's SAE
code vectors on training images in the union support. Variance is the mean
population variance across the selected latent dimensions; cosine similarity is
the mean across distinct image pairs. A group with fewer than two nonzero code
vectors has an empty cosine value. Both metrics are averaged by neighborhood
size in `summary.csv`.

Every accuracy-drop result also has a probability-drop result. It measures the
decrease in the classifier's softmax probability for each image's true class
relative to the unmodified SAE reconstruction. Target and per-class values are
means over their class images, non-target values average the other class means,
and overall values average all test images. `summary.csv` includes matching
probability means and size-1 changes.

Results are written to `SAE/<sae-run>/steering/<timestamp>/`:

```text
neighborhoods.csv   # each anchor and size: latents, drops, support, representation and code statistics
summary.csv         # means over all anchors and over anchors with nonzero size-1 target drop
metadata.json       # settings, source runs, class mapping, baseline accuracies
classifier.pt       # fixed linear classifier weights
latent_statistics.pt # sampled anchor similarity columns, means and support counts
```

If the source image run has moved, pass `--model-log-dir` to its new path. The
evaluation uses the saved exported splits; if `sae_smples` sampled them in the
image run, the classifier and metrics use those sampled images.

## Most activated images

The pipeline finishes by ranking train and test images for every SAE latent and
every size-2-or-larger group in the selected steering run. You can also run this
stage separately:

```bash
python main_sae_inference.py \
  --sae-log-dir log_dir/<image-run>/SAE/<sae-run> \
  --steering-log-dir log_dir/<image-run>/SAE/<sae-run>/steering/<steering-run> \
  --top-active-images 3
```

Set `top_active_images` in `INFERENCE_CONFIG` in `cfgs/cfg_sae.py`, through the
pipeline `inference` overrides, or on the command line. A value of 3 makes a
3×3 grid of up to nine images. Each `active_images/train/` and
`active_images/test/` directory contains a `.txt` ranking and `.png` grid for
each latent and group. The title names the latent or lists the group's members;
each tile displays its activation. A group scores an image by summing its member
SAE activations. Only images with a positive score are shown, and ties preserve
the saved feature order. BLIP/DINO runs use their full cached feature splits;
other runs use the exported SAE embedding splits, which may be sampled by
`sae_smples`. If `--steering-log-dir` is omitted, the latest steering run for
that SAE is used.

Run the integration checks with `python -m unittest discover -s tests -v`.
