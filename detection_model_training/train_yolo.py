#!/usr/bin/env python3
"""Generic YOLO11 detector trainer for the project's detector datasets.

Place this file directly in:
    detection_model_training/

It can train either YOLO11-nano or YOLO11-small on any of the three current
experiment datasets:
    easy   -> datasets/easy
    medium -> datasets/medium
    hard   -> datasets/hard

The default training/evaluation image size is 320, matching the Webots camera
width and the deployed YOLO inference setting. Training keeps Ultralytics'
normal aspect-ratio-preserving preprocessing; the source 320x240 frames are not
stretched to 320x320.

Generated training and evaluation artifacts are placed in detector-specific
folders under `detection_model_training/`. Final deployment checkpoints are
published separately under `detection_model_training/checkpoints/`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from eval_yolo import (
    ENV_ALIASES,
    MODEL_ALIASES,
    PROJECT_ROOT,
    build_paths,
    canonical_env,
    canonical_model_size,
    evaluate_checkpoint,
    repo_path,
    write_runtime_dataset_yaml,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO11 nano on easy, medium, or hard data")
    p.add_argument("--env", default="medium", choices=sorted(ENV_ALIASES))
    p.add_argument("--model-size", default="nano", choices=sorted(MODEL_ALIASES),
                   help="YOLO11 variant: nano (n)")
    p.add_argument("--model", default=None,
                   help="Optional base checkpoint override. By default uses root yolo11n.pt.")
    p.add_argument("--data", default=None,
                   help="Optional dataset YAML override. Relative paths resolve from detection_model_training/.")
    p.add_argument("--project", default=None,
                   help="Optional Ultralytics training project directory override.")
    p.add_argument("--name", default=None, help="Optional training run-name override.")
    p.add_argument("--eval-project", default=None, help="Optional final evaluation output override.")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=320,
                   help="YOLO target input size. 320 matches the deployed Webots detector.")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rect", action=argparse.BooleanOptionalAction, default=False,
                   help="Use Ultralytics rectangular training. Default False preserves the previous recipe.")
    p.add_argument("--exist-ok", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--final-eval", action=argparse.BooleanOptionalAction, default=True,
                   help="Evaluate best.pt on val+test after training.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env_name = canonical_env(args.env)
    model_size = canonical_model_size(args.model_size)

    paths = build_paths(env_name, model_size)
    paths.model_root.mkdir(parents=True, exist_ok=True)

    data_yaml = repo_path(args.data) if args.data else write_runtime_dataset_yaml(env_name, paths.model_root)
    project = repo_path(args.project) if args.project else paths.train_project
    run_name = args.name or paths.run_name
    eval_project = repo_path(args.eval_project) if args.eval_project else paths.eval_project

    if args.model:
        model_weights = args.model
    else:
        local_base = PROJECT_ROOT / paths.base_checkpoint
        model_weights = str(local_base) if local_base.exists() else paths.base_checkpoint

    print(f"Project root:     {PROJECT_ROOT}")
    print(f"Environment:      {env_name}")
    print(f"YOLO variant:     {model_size}")
    print(f"Training YAML:    {data_yaml}")
    print(f"Base model:       {model_weights}")
    print(f"Training imgsz:   {args.imgsz}")
    print(f"Rect training:    {args.rect}")
    print(f"Run dir target:   {project / run_name}")
    print(f"Eval project:     {eval_project}")

    if args.imgsz != 320:
        print(
            f"\n[WARNING] imgsz={args.imgsz}; the Webots deployment path uses imgsz=320. "
            "Use a non-320 value only intentionally.",
            flush=True,
        )

    print("\nTraining YAML contents:")
    print(data_yaml.read_text(encoding="utf-8"))

    # Keep the latest YOLO-small hard recipe unchanged except for:
    #   1) selected YOLO11 variant,
    #   2) selected dataset/output paths,
    #   3) imgsz default changed from 640 -> 320 to match deployment.
    train_args = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        pretrained=True,
        optimizer="auto",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        patience=args.patience,
        cos_lr=False,
        amp=args.amp,
        cache=args.cache,
        freeze=0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        rect=args.rect,
        project=str(project),
        name=run_name,
        exist_ok=args.exist_ok,
        verbose=True,
        plots=True,
        save=True,
        save_period=-1,
        val=True,
    )

    model = YOLO(model_weights)
    model.train(**train_args)

    if getattr(model, "trainer", None) is not None:
        actual_run_dir = Path(model.trainer.save_dir).resolve()
    else:
        actual_run_dir = (project / run_name).resolve()

    best_weights = actual_run_dir / "weights" / "best.pt"
    last_weights = actual_run_dir / "weights" / "last.pt"

    print("\nTraining finished.")
    print(f"Actual run dir:   {actual_run_dir}")
    print(f"Best weights:     {best_weights}")
    print(f"Last weights:     {last_weights}")

    if not best_weights.exists():
        raise FileNotFoundError(f"Training finished but best.pt was not found: {best_weights}")

    if args.final_eval:
        evaluate_checkpoint(
            env=env_name,
            model_size=model_size,
            weights=best_weights,
            data=data_yaml,
            eval_project=eval_project,
            split="both",
            device=str(args.device),
            imgsz=int(args.imgsz),
        )


if __name__ == "__main__":
    main()
