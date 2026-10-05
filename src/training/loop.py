"""Training loop: teacher forcing (+ optional scheduled sampling), label smoothing, early stopping."""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from ..evaluation.evaluate import evaluate_rows
from ..inference.generate import translate_texts
from ..models.factory import save_checkpoint
from ..tokenization.base import PAD_ID
from ..utils.logging import ExperimentTracker, get_logger

log = get_logger()


def make_optimizer(model, cfg):
    kind = cfg.get("optimizer", "adamw").lower()
    params = model.parameters()
    if kind == "adamw":
        return torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    if kind == "adam":
        return torch.optim.Adam(params, lr=cfg["lr"])
    if kind == "sgd":
        return torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9)
    raise ValueError(kind)


def make_scheduler(optimizer, cfg, steps_per_epoch: int):
    kind = cfg.get("scheduler", "plateau")
    if kind == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                           patience=max(1, cfg.get("patience", 8) // 2))
    if kind == "warmup_inv_sqrt":
        warmup = max(1, cfg.get("warmup_steps", 400))
        fn = lambda step: min((step + 1) / warmup, (warmup / (step + 1)) ** 0.5)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)
    return None


def _tf_ratio(epoch: int, total: int, start: float, end: float) -> float:
    if total <= 1:
        return end
    return start + (end - start) * (epoch / (total - 1))


def run_epoch(model, loader, optimizer, scheduler, criterion, device, grad_clip, tf_ratio, train: bool, use_amp: bool = False):
    model.train(train)
    total_loss, total_tok = 0.0, 0
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None
    for batch in loader:
        src, src_len = batch["src"].to(device), batch["src_len"]
        tin, tout = batch["tgt_in"].to(device), batch["tgt_out"].to(device)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(src, src_len, tin, teacher_forcing_ratio=tf_ratio if train else 1.0)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tout.reshape(-1))
        n_tok = (tout != PAD_ID).sum().item()
        if train:
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
        total_loss += loss.item() * n_tok
        total_tok += n_tok
    return total_loss / max(total_tok, 1)


def train_model(model, src_tok, tgt_tok, train_loader, valid_rows, cfg_train, device="cpu", run_dir=None):
    run_dir = Path(run_dir or Path(cfg_train["out_dir"]) / cfg_train["run_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg_train.get("log_dir", "experiments/logs")) / cfg_train["run_name"]
    tracker = ExperimentTracker(log_dir, cfg_train.get("tensorboard", False),
                                cfg_train.get("wandb", False), run_name=cfg_train["run_name"])
    optimizer = make_optimizer(model, cfg_train)
    scheduler = make_scheduler(optimizer, cfg_train, len(train_loader))
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=cfg_train.get("label_smoothing", 0.0))
    monitor = cfg_train.get("monitor", "valid_cer")
    best_score, best_epoch, bad_epochs = math.inf, -1, 0
    history = []
    model.to(device)
    t0 = time.time()
    for epoch in range(cfg_train["epochs"]):
        train_loader.set_epoch(epoch)
        tf = _tf_ratio(epoch, cfg_train["epochs"], cfg_train.get("tf_ratio_start", 1.0), cfg_train.get("tf_ratio_end", 1.0))
        use_amp = cfg_train.get('mixed_precision', False) and str(device).startswith('cuda')
        train_loss = run_epoch(model, train_loader, optimizer, scheduler, criterion, device, cfg_train.get("grad_clip", 1.0), tf, True, use_amp=use_amp)
        metrics = {"epoch": epoch, "train_loss": train_loss, "train_ppl": math.exp(min(train_loss, 20)), "tf_ratio": tf,
                   "lr": optimizer.param_groups[0]["lr"], "elapsed_s": round(time.time() - t0, 1)}
        do_eval = (epoch + 1) % cfg_train.get("eval_every", 1) == 0 or epoch == cfg_train["epochs"] - 1
        if do_eval:
            sample = valid_rows[: cfg_train.get("eval_samples", len(valid_rows))]
            preds = [h[0]["text"] for h in translate_texts(model, src_tok, tgt_tok, [r["src"] for r in sample], device)]
            ev = evaluate_rows(sample, preds, slices=False)["overall"]
            metrics.update({f"valid_{k}": v for k, v in ev.items() if isinstance(v, (int, float))})
            score = metrics.get(monitor, metrics["valid_cer"])
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(score)
            improved = score < best_score - 1e-4
            if improved:
                best_score, best_epoch, bad_epochs = score, epoch, 0
                save_checkpoint(run_dir / "best.pt", model, model.__dict__.get("_mk", {}), src_tok, tgt_tok,
                                meta={"epoch": epoch, "monitor": monitor, "score": score}, epoch=epoch)
            else:
                bad_epochs += 1
            metrics["best_" + monitor] = best_score
            log.info("epoch %d | loss %.4f | %s %.4f | best %.4f (ep %d) | tf %.2f",
                     epoch, train_loss, monitor, score, best_score, best_epoch, tf)
        else:
            log.info("epoch %d | loss %.4f | tf %.2f", epoch, train_loss, tf)
        tracker.log(epoch, metrics)
        history.append(metrics)
        save_checkpoint(run_dir / "last.pt", model, model.__dict__.get("_mk", {}), src_tok, tgt_tok,
                        meta={"epoch": epoch}, optimizer=optimizer, epoch=epoch)
        if do_eval and bad_epochs >= cfg_train.get("patience", 8):
            log.info("Early stopping at epoch %d (no improvement in %d evals).", epoch, bad_epochs)
            break
    tracker.close()
    return {"history": history, "best_epoch": best_epoch, "best_score": best_score, "run_dir": str(run_dir)}
