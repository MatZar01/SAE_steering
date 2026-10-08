"""Point 1: evaluate SAE ablation over support-based latent neighborhoods."""

import csv
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from cfgs.cfg_sae import SteeringConfig
from src.models_sae import load_sae, resolve_sae_run
from src.trainer import create_run_dir


RESULT_FIELDS = (
    "anchor", "size", "latents", "target_class", "target_class_index",
    "target_accuracy_drop", "non_target_mean_accuracy_drop", "overall_accuracy_drop",
    "support_size", "mean_per_dimension_variance", "total_variance",
    "mean_squared_distance_to_centroid", "class_accuracy_drops",
)
SUMMARY_FIELDS = (
    "population", "size", "anchors", "mean_target_accuracy_drop",
    "mean_non_target_accuracy_drop", "mean_overall_accuracy_drop",
    "mean_support_size", "mean_per_dimension_variance", "mean_total_variance",
    "mean_squared_distance_to_centroid", "target_drop_change_from_size_1",
    "non_target_drop_change_from_size_1", "total_variance_change_from_size_1",
)


def _device(name: str) -> torch.device:
    if name == "auto":
        name = ("cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def _load_split(source: Path, split: str, num_classes: int):
    out = source / "out"
    embeddings = torch.load(out / f"embeddings_{split}.pt", map_location="cpu", weights_only=True)
    labels = torch.load(out / f"labels_{split}.pt", map_location="cpu", weights_only=True)
    if (not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2
            or not embeddings.is_floating_point() or not torch.isfinite(embeddings).all()):
        raise ValueError(f"{split} embeddings must be a finite float tensor [N, D].")
    if (not isinstance(labels, torch.Tensor) or labels.ndim != 1
            or labels.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
            or len(labels) != len(embeddings) or len(labels) == 0):
        raise ValueError(f"{split} labels must be an integer tensor [N] aligned with embeddings.")
    labels = labels.long()
    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError(f"{split} labels are outside the class_to_idx.json range.")
    return embeddings.float().contiguous(), labels.contiguous()


def train_linear_classifier(embeddings: torch.Tensor, labels: torch.Tensor,
                            num_classes: int, config: SteeringConfig,
                            device: torch.device) -> nn.Linear:
    """Fit one affine layer on the original training representations."""
    torch.manual_seed(config.seed)
    classifier = nn.Linear(embeddings.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=config.classifier_learning_rate,
                                  weight_decay=config.classifier_weight_decay)
    generator = torch.Generator().manual_seed(config.seed)
    classifier.train()
    for _ in range(config.classifier_epochs):
        indices = torch.randperm(len(embeddings), generator=generator)
        for batch in indices.split(config.classifier_batch_size):
            logits = classifier(embeddings[batch].to(device))
            loss = F.cross_entropy(logits, labels[batch].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return classifier.eval().requires_grad_(False)


@torch.inference_mode()
def _correct_counts(classifier: nn.Module, embeddings: torch.Tensor,
                    labels: torch.Tensor, batch_size: int, num_classes: int,
                    device: torch.device) -> torch.Tensor:
    correct = torch.zeros(num_classes, dtype=torch.long)
    for start in range(0, len(embeddings), batch_size):
        end = start + batch_size
        prediction = classifier(embeddings[start:end].to(device)).argmax(dim=1).cpu()
        correct += torch.bincount(labels[start:end][prediction == labels[start:end]],
                                  minlength=num_classes)
    return correct


@torch.inference_mode()
def _training_support(sae: nn.Module, embeddings: torch.Tensor,
                      batch_size: int, device: torch.device):
    """Use the full exported training set for means, activity, and strict supports."""
    num_latents = sae.nb_concepts
    activation_sum = torch.zeros(num_latents, dtype=torch.float64)
    active_count = torch.zeros(num_latents, dtype=torch.long)
    for batch in embeddings.split(batch_size):
        codes = sae.encode(batch.to(device))[1].cpu()
        if not torch.isfinite(codes).all():
            raise ValueError("SAE produced non-finite training codes.")
        activation_sum += codes.sum(dim=0, dtype=torch.float64)
        active_count += (codes > 0).sum(dim=0)
    mean_activation = activation_sum / len(embeddings)
    support = torch.empty((len(embeddings), num_latents), dtype=torch.bool)
    for start in range(0, len(embeddings), batch_size):
        end = min(start + batch_size, len(embeddings))
        codes = sae.encode(embeddings[start:end].to(device))[1].cpu()
        support[start:end] = codes > mean_activation
    return mean_activation, active_count, support


def _similarity_to_anchors(support: torch.Tensor, anchors: torch.Tensor,
                           penalty: float, batch_size: int,
                           device: torch.device) -> torch.Tensor:
    """Symmetric Tversky overlap, computed only for sampled anchor columns."""
    counts = support.sum(dim=0).double()
    intersection = torch.zeros((support.shape[1], len(anchors)),
                               dtype=torch.float64)
    anchor_columns = anchors.to(device)
    for batch in support.split(batch_size):
        values = batch.to(device=device, dtype=torch.float32)
        intersection += (values.T @ values[:, anchor_columns]).cpu().double()
    exclusive = counts[:, None] + counts[anchors][None, :] - 2 * intersection
    denominator = intersection + penalty * exclusive
    return intersection / denominator.masked_fill(denominator == 0, 1)


@torch.inference_mode()
def _test_codes_and_baseline(sae: nn.Module, classifier: nn.Module,
                             embeddings: torch.Tensor, labels: torch.Tensor,
                             batch_size: int, num_classes: int,
                             device: torch.device):
    codes = torch.empty((len(embeddings), sae.nb_concepts), dtype=torch.float32)
    reconstructions = torch.empty_like(embeddings)
    baseline_correct = torch.zeros(num_classes, dtype=torch.long)
    for start in range(0, len(embeddings), batch_size):
        end = min(start + batch_size, len(embeddings))
        batch_codes = sae.encode(embeddings[start:end].to(device))[1]
        if not torch.isfinite(batch_codes).all():
            raise ValueError("SAE produced non-finite evaluation codes.")
        codes[start:end] = batch_codes.cpu()
        reconstructed = sae.decode(batch_codes)
        reconstructions[start:end] = reconstructed.cpu()
        prediction = classifier(reconstructed).argmax(dim=1).cpu()
        batch_labels = labels[start:end]
        baseline_correct += torch.bincount(batch_labels[prediction == batch_labels],
                                           minlength=num_classes)
    return codes, reconstructions, baseline_correct


@torch.inference_mode()
def _ablated_correct_by_size(classifier: nn.Module, codes: torch.Tensor,
                             reconstructions: torch.Tensor, dictionary: torch.Tensor,
                             labels: torch.Tensor, neighborhood: list[int],
                             batch_size: int, num_classes: int,
                             device: torch.device) -> torch.Tensor:
    """Subtract each new latent's exact linear decoder contribution."""
    correct = torch.zeros((len(neighborhood), num_classes), dtype=torch.long)
    for start in range(0, len(codes), batch_size):
        end = min(start + batch_size, len(codes))
        reconstructed = reconstructions[start:end].to(device).clone()
        batch_labels = labels[start:end]
        for size, latent in enumerate(neighborhood):
            reconstructed -= codes[start:end, latent].to(device)[:, None] * dictionary[latent]
            prediction = classifier(reconstructed).argmax(dim=1).cpu()
            correct[size] += torch.bincount(batch_labels[prediction == batch_labels],
                                             minlength=num_classes)
    return correct


def _neighborhood_variance(embeddings: torch.Tensor, support: torch.Tensor,
                           neighborhood: list[int]):
    """Accumulate moments as nested support unions grow."""
    included = torch.zeros(len(embeddings), dtype=torch.bool)
    count = 0
    total = torch.zeros(embeddings.shape[1], dtype=torch.float64)
    squared_total = torch.zeros_like(total)
    for latent in neighborhood:
        new = support[:, latent] & ~included
        included |= new
        if new.any():
            values = embeddings[new].double()
            count += len(values)
            total += values.sum(dim=0)
            squared_total += values.square().sum(dim=0)
        mean = total / count
        per_dimension_variance = (squared_total / count - mean.square()).clamp_min(0)
        yield (count, per_dimension_variance.mean().item(),
               per_dimension_variance.sum().item())


def _summaries(rows: list[dict], max_size: int) -> list[dict]:
    at_one = {row["anchor"]: row for row in rows if row["size"] == 1}
    nonzero = {anchor for anchor, row in at_one.items()
               if abs(row["target_accuracy_drop"]) > 1e-12}
    summaries = []
    for population, selected in (("all", set(at_one)), ("nonzero_size_1", nonzero)):
        for size in range(1, max_size + 1):
            group = [row for row in rows if row["size"] == size and row["anchor"] in selected]
            summary = {"population": population, "size": size, "anchors": len(group)}
            metrics = (
                ("mean_target_accuracy_drop", "target_accuracy_drop"),
                ("mean_non_target_accuracy_drop", "non_target_mean_accuracy_drop"),
                ("mean_overall_accuracy_drop", "overall_accuracy_drop"),
                ("mean_support_size", "support_size"),
                ("mean_per_dimension_variance", "mean_per_dimension_variance"),
                ("mean_total_variance", "total_variance"),
                ("mean_squared_distance_to_centroid", "mean_squared_distance_to_centroid"),
            )
            for output_name, row_name in metrics:
                summary[output_name] = statistics.fmean(row[row_name] for row in group) if group else None
            for output_name, row_name in (
                ("target_drop_change_from_size_1", "target_accuracy_drop"),
                ("non_target_drop_change_from_size_1", "non_target_mean_accuracy_drop"),
                ("total_variance_change_from_size_1", "total_variance"),
            ):
                summary[output_name] = (statistics.fmean(row[row_name] - at_one[row["anchor"]][row_name]
                                                          for row in group) if group else None)
            summaries.append(summary)
    return summaries


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_steering(config: SteeringConfig) -> Path:
    """Run the PDF's neighborhood ablation evaluation on one completed SAE."""
    config.validate()
    sae_run = resolve_sae_run(config.resolve(config.sae_log_dir))
    device = _device(config.device)
    sae, artifact = load_sae(sae_run, device)
    source = (config.resolve(config.model_log_dir) if config.model_log_dir else
              (sae_run / artifact["source_model_relative_path"]).resolve())
    if not (source / "out/embeddings_train.pt").is_file() and not config.model_log_dir:
        source = Path(artifact["source_model_log_dir"])
    class_to_idx = json.loads((source / "class_to_idx.json").read_text())
    if sorted(class_to_idx.values()) != list(range(len(class_to_idx))) or len(class_to_idx) < 2:
        raise ValueError("class_to_idx.json must contain at least two contiguous classes.")
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    num_classes = len(class_names)
    train_embeddings, train_labels = _load_split(source, "train", num_classes)
    test_embeddings, test_labels = _load_split(source, "test", num_classes)
    if train_embeddings.shape[1] != artifact["architecture"]["input_dim"]:
        raise ValueError("SAE input dimension differs from training embeddings.")
    if test_embeddings.shape[1] != train_embeddings.shape[1]:
        raise ValueError("Train and test embedding dimensions differ.")
    train_counts = torch.bincount(train_labels, minlength=num_classes)
    test_counts = torch.bincount(test_labels, minlength=num_classes)
    if (train_counts == 0).any():
        raise ValueError("Exported training embeddings must include every class to fit the classifier.")
    observed_classes = torch.nonzero(test_counts > 0).flatten().tolist()
    if len(observed_classes) < 2:
        raise ValueError("Evaluation embeddings must contain at least two classes.")

    classifier = train_linear_classifier(train_embeddings, train_labels,
                                         num_classes, config, device)
    train_original = _correct_counts(classifier, train_embeddings, train_labels,
                                     config.evaluation_batch_size, num_classes, device)
    test_original = _correct_counts(classifier, test_embeddings, test_labels,
                                    config.evaluation_batch_size, num_classes, device)
    mean_activation, active_count, support = _training_support(
        sae, train_embeddings, config.evaluation_batch_size, device)
    support_counts = support.sum(dim=0)
    eligible = torch.nonzero((active_count > 0) & (support_counts > 0)).flatten()
    if len(eligible) == 0:
        raise ValueError("No SAE latent has nonempty training support.")
    if config.max_anchors is not None and len(eligible) > config.max_anchors:
        generator = torch.Generator().manual_seed(config.seed)
        anchors = eligible[torch.randperm(len(eligible), generator=generator)[:config.max_anchors]].sort().values
    else:
        anchors = eligible
    max_size = min(config.max_neighborhood_size, len(eligible))
    similarity = _similarity_to_anchors(support, anchors, config.tversky_penalty,
                                        config.evaluation_batch_size, device)
    test_codes, reconstructions, baseline_correct = _test_codes_and_baseline(
        sae, classifier, test_embeddings, test_labels,
        config.evaluation_batch_size, num_classes, device)
    dictionary = sae.get_dictionary().detach()
    baseline_accuracy = baseline_correct.double() / test_counts.clamp_min(1)
    baseline_overall = baseline_correct.sum().item() / len(test_labels)

    rows = []
    for column, anchor_tensor in enumerate(anchors):
        anchor = int(anchor_tensor)
        scores = similarity[:, column].clone()
        scores[support_counts == 0] = -float("inf")
        scores[anchor] = -float("inf")
        ranked = torch.argsort(scores, descending=True, stable=True)
        neighborhood = [anchor] + ranked[:max_size - 1].tolist()
        correct_by_size = _ablated_correct_by_size(
            classifier, test_codes, reconstructions, dictionary, test_labels,
            neighborhood, config.evaluation_batch_size, num_classes, device)
        first_drop = baseline_accuracy - correct_by_size[0].double() / test_counts.clamp_min(1)
        target = min(observed_classes, key=lambda idx: (-first_drop[idx].item(), idx))
        for size, (correct, variance) in enumerate(zip(correct_by_size,
                _neighborhood_variance(train_embeddings, support, neighborhood)), start=1):
            drops = baseline_accuracy - correct.double() / test_counts.clamp_min(1)
            support_size, mean_variance, total_variance = variance
            rows.append({
                "anchor": anchor, "size": size,
                "latents": "|".join(map(str, neighborhood[:size])),
                "target_class": class_names[target], "target_class_index": target,
                "target_accuracy_drop": drops[target].item(),
                "non_target_mean_accuracy_drop": statistics.fmean(
                    drops[idx].item() for idx in observed_classes if idx != target),
                "overall_accuracy_drop": baseline_overall - correct.sum().item() / len(test_labels),
                "support_size": support_size,
                "mean_per_dimension_variance": mean_variance,
                "total_variance": total_variance,
                # Population variance identity: E||h - mean(h)||^2 = sum_d Var(h_d).
                "mean_squared_distance_to_centroid": total_variance,
                "class_accuracy_drops": json.dumps({class_names[idx]: drops[idx].item()
                                                     for idx in observed_classes}),
            })
        if (column + 1) % max(1, len(anchors) // 20) == 0 or column + 1 == len(anchors):
            print(f"Evaluated anchor {column + 1}/{len(anchors)} (latent {anchor})", flush=True)

    summary = _summaries(rows, max_size)
    run_dir = create_run_dir(sae_run / "steering")
    _write_csv(run_dir / "neighborhoods.csv", RESULT_FIELDS, rows)
    _write_csv(run_dir / "summary.csv", SUMMARY_FIELDS, summary)
    torch.save({"state_dict": {key: value.cpu() for key, value in classifier.state_dict().items()},
                "input_dim": train_embeddings.shape[1], "num_classes": num_classes},
               run_dir / "classifier.pt")
    torch.save({"anchor_ids": anchors, "similarity": similarity.float(),
                "mean_activation": mean_activation.float(),
                "active_count": active_count, "support_count": support_counts},
               run_dir / "latent_statistics.pt")
    metadata = {
        "config": config.to_dict(), "sae_run": str(sae_run), "source_model_log_dir": str(source),
        "class_to_idx": class_to_idx, "num_train": len(train_labels), "num_test": len(test_labels),
        "num_non_dead_latents": int((active_count > 0).sum()),
        "num_eligible_latents": len(eligible), "num_sampled_anchors": len(anchors),
        "max_evaluated_size": max_size,
        "similarity": "|A∩B| / (|A∩B| + penalty*(|A\\B| + |B\\A|))",
        "support": "z_k(x) > mean_train(z_k)",
        "accuracy_drop_baseline": "unmodified SAE reconstruction on exported test embeddings",
        "nonzero_size_1_anchors": sum(abs(row["target_accuracy_drop"]) > 1e-12
                                      for row in rows if row["size"] == 1),
        "classifier_original_train_accuracy": train_original.sum().item() / len(train_labels),
        "classifier_original_test_accuracy": test_original.sum().item() / len(test_labels),
        "classifier_reconstructed_test_accuracy": baseline_overall,
        "classifier_original_test_accuracy_by_class": {
            class_names[idx]: test_original[idx].item() / test_counts[idx].item()
            for idx in observed_classes},
        "classifier_reconstructed_test_accuracy_by_class": {
            class_names[idx]: baseline_accuracy[idx].item() for idx in observed_classes},
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return run_dir
