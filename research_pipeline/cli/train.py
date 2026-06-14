from __future__ import annotations

import argparse

from research_pipeline.cli.common import load_yaml, project_path
from research_pipeline.models.artifacts import save_artifact
from research_pipeline.models.classical import train_random_forest
from research_pipeline.models.temporal import train_temporal_prototype
from research_pipeline.models.torch_training import train_tcn
from research_pipeline.utils.errors import PipelineError


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a recognizer from a YAML config.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    model_type = config.get("model_type", "temporal_prototype")
    manifest = project_path(config["manifest"])
    output_model = project_path(config["output_model"])
    seed = int(config.get("seed", 13))
    target_length = int(config.get("target_length", 32))
    validation_manifest = config.get("validation_manifest")
    channels = config.get("tcn_channels", config.get("channels"))

    if model_type == "c0_rule":
        artifact = {
            "model_type": "c0_rule",
            "target_length": target_length,
            "seed": seed,
            "params": config.get("params", {}),
        }
        save_artifact(output_model, artifact)
    elif model_type == "c1_random_forest":
        artifact = train_random_forest(
            manifest,
            output_model,
            seed=seed,
            target_length=target_length,
            n_estimators=int(config.get("n_estimators", 200)),
        )
    elif model_type == "c1t_tcn" and config.get("backend", "auto") == "torch":
        artifact = train_tcn(
            manifest,
            output_model,
            seed=seed,
            target_length=target_length,
            epochs=int(config.get("epochs", 40)),
            batch_size=int(config.get("batch_size", 32)),
            learning_rate=float(config.get("learning_rate", 1e-3)),
            weight_decay=float(config.get("weight_decay", 1e-2)),
            validation_manifest_path=project_path(validation_manifest) if validation_manifest else None,
            validation_split=float(config.get("validation_split", 0.0)),
            early_stopping_patience=int(config.get("early_stopping_patience", 0)),
            early_stopping_min_delta=float(config.get("early_stopping_min_delta", 0.0)),
            channels=tuple(int(value) for value in channels) if channels else None,
            kernel_size=int(config.get("kernel_size", 3)),
            dropout=float(config.get("dropout", 0.15)),
            pooling=str(config.get("pooling", "avg")),
            balanced_sampler=bool(config.get("balanced_sampler", False)),
            focal_gamma=float(config.get("focal_gamma", 0.0)),
            label_smoothing=float(config.get("label_smoothing", 0.0)),
            augmentation=config.get("augmentation", {}),
        )
    elif model_type in {"temporal_prototype", "c1t_auto", "c1t_tcn"}:
        artifact = train_temporal_prototype(manifest, output_model, seed=seed, target_length=target_length)
    else:
        raise PipelineError(f"Unsupported model_type '{model_type}'.")
    print(f"trained {artifact['model_type']} -> {output_model}")


if __name__ == "__main__":
    main()
