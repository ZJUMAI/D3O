import json
import copy
from collections import defaultdict
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from ordinalclip.models import MODELS
from ordinalclip.utils.logging import get_logger

from .optim import build_lr_scheduler, build_optimizer, build_staged_lr_param_groups
from .utils import freeze_param, load_pretrained_weights

import os
import matplotlib.pyplot as plt
import pandas as pd

logger = get_logger('__main__')


class Runner(pl.LightningModule):
    def __init__(
        self,
        model_cfg,
        output_dir: str,
        optimizer_and_scheduler_cfg,
        load_weights_cfg,
        seed: int,
        teacher_mode: str = "ema",
        conle_alpha: float = 0.1,
        conle_beta: float = 0.01,
        conle_tau: float = 0.5,
        conle_threshold: float = 0.05,
        loss_weights=dict(
            ce_loss=1.0,
            kl_loss=1.0,
            reg_loss=1.0,
            kd_loss=1.0,
            kd_temp=2.0,
            feat_kd_loss=1.0,
            shallow_kd_loss=1.0,
            shallow_ce_loss=1.0,
            conle_loss=1.0,
        ),
        ckpt_path="",
    ) -> None:
        super().__init__()

        self.module = MODELS.build(model_cfg)
        # Configurable label smoothing for CE loss
        self.ce_loss_func = nn.CrossEntropyLoss()
        # self.ce_loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.kl_loss_func = nn.KLDivLoss(reduction="sum")
        # self.reg_loss_func = nn.L1Loss()
        self.reg_loss_func = nn.MSELoss()

        # Merge user-provided weights with defaults so newly introduced keys
        # (e.g., shallow_ce_loss) work even if older YAMLs don't include them.
        default_loss_weights = {
            "ce_loss": 1.0,
            "kl_loss": 1.0,
            "reg_loss": 1.0,
            "kd_loss": 1.0,
            "kd_temp": 2.0,
            "feat_kd_loss": 1.0,
            "shallow_kd_loss": 0.0,
            "shallow_ce_loss": 1.0,
            "conle_loss": 1.0,
        }
        user_loss_weights = dict(loss_weights) if loss_weights is not None else {}
        default_loss_weights.update(user_loss_weights)
        self.loss_weights = default_loss_weights
        print(f"Using loss weights: {self.loss_weights}")
        self.kd_temp = float(self.loss_weights.get("kd_temp", 2.0))

        self.teacher_mode = str(teacher_mode).lower().strip()
        print(f"Using teacher mode: {self.teacher_mode}" )

        # Contrastive Label Enhancement (ConLE) hyper-params (not part of loss_weights).
        self.conle_alpha = float(conle_alpha)
        self.conle_beta = float(conle_beta)
        self.conle_tau = float(conle_tau)
        self.conle_threshold = float(conle_threshold)
        self.num_ranks = self.module.num_ranks
        # self.num_ranks = 5
        self.num_class = self.num_ranks
        self.register_buffer("rank_output_value_array", torch.arange(0, self.num_ranks).float(), persistent=False)
        self.output_dir = Path(output_dir)
        # self._custom_logger = get_logger(__name__)
        self._custom_logger = logger


        self.load_weights(**load_weights_cfg)

        # EMA teacher for self-distillation. Teacher is inference-only and updated
        # solely via EMA(student) after each student optimizer step.
        self.ema_decay = float(getattr(self.module, "ema_decay", 0.995))
        self.teacher = copy.deepcopy(self.module)
        if self.teacher_mode != "ema":
            raise ValueError(
                f"Unsupported teacher_mode: {teacher_mode}. This runner supports EMA-only teacher (teacher_mode='ema')."
            )
        self.teacher.eval()
        self.teacher.requires_grad_(False)

        self._optimizer_and_scheduler_cfg = optimizer_and_scheduler_cfg
        self.seed = seed
        self.ckpt_path = ckpt_path
        self.acc = 0
        self._track10_history_rows = []

    # Model Forward
    def forward(self, images):
        return self.module(images)

    def forward_text_only(self):
        return self.module.forward_text_only()

    def aggregate_loss(self, losses: dict):
        total = None
        for k, weight in self.loss_weights.items():
            if k == "kd_temp":
                continue
            if k in losses:
                term = float(weight) * losses[k]
                total = term if total is None else (total + term)

        if total is None:
            # Return a device-correct zero tensor so Lightning can backprop safely.
            device = None
            dtype = None
            for v in losses.values():
                if torch.is_tensor(v):
                    device = v.device
                    dtype = v.dtype
                    break
            if device is None:
                device = getattr(self, "device", None)
            if device is None:
                try:
                    device = next(self.parameters()).device
                except StopIteration:
                    device = torch.device("cpu")
            if dtype is None:
                dtype = torch.float32
            total = torch.zeros((), device=device, dtype=dtype)

        return total

    def _unpack_model_output(self, res):
        """Normalize model outputs across different datasets/models.

        Returns:
            logits, regression_age, image_features, text_features, patch_features, aux
        """
        aux = None
        patch_features = None

        if isinstance(res, (tuple, list)):
            if len(res) == 6:
                logits, regression_age, image_features, text_features, patch_features, aux = res
            elif len(res) == 5:
                logits, regression_age, image_features, text_features, patch_features = res
            elif len(res) == 4:
                logits, regression_age, image_features, text_features = res
            else:
                raise ValueError(f"Unexpected model output tuple length: {len(res)}")
        else:
            # logits-only output
            logits = res
            regression_age = 0
            image_features = None
            text_features = None

        return logits, regression_age, image_features, text_features, patch_features, aux

    @torch.no_grad()
    def _collect_fixed_tracking_enhance_logits(self):
        trainer = getattr(self, "trainer", None)
        if trainer is None or getattr(trainer, "datamodule", None) is None:
            return None
        train_set = getattr(trainer.datamodule, "train_set", None)
        if train_set is None or not hasattr(train_set, "get_tracking_samples"):
            return None

        fixed = train_set.get_tracking_samples()
        if fixed is None:
            return None
        if not isinstance(fixed, dict):
            return None
        if "images" not in fixed or "labels" not in fixed or "indices" not in fixed:
            return None

        x_fixed = fixed["images"].to(self.device, non_blocking=True)
        res = self.module(x_fixed)
        _, _, _, _, _, aux = self._unpack_model_output(res)
        if not isinstance(aux, dict):
            return None

        enhance_logits = aux.get("enhance_logits", None)
        if enhance_logits is None:
            return None

        sample_indices = list(fixed["indices"])
        image_paths = []
        images_file = getattr(train_set, "images_file", None)
        images_root = getattr(train_set, "images_root", None)
        if isinstance(images_file, list):
            for idx in sample_indices:
                idx_int = int(idx)
                if 0 <= idx_int < len(images_file):
                    rel_path = str(images_file[idx_int])
                    if images_root:
                        rel_path = rel_path if os.path.isabs(rel_path) else os.path.join(str(images_root), rel_path)
                    image_paths.append(rel_path)
                else:
                    image_paths.append("")
        else:
            image_paths = [""] * len(sample_indices)

        return {
            "sample_indices": torch.tensor(sample_indices, dtype=torch.long, device=self.device),
            "labels": fixed["labels"].to(self.device, non_blocking=True),
            "enhance_logits": enhance_logits,
            "image_paths": image_paths,
        }

    def compute_feat_kd_loss(self, student_aux, teacher_aux):
        if not isinstance(student_aux, dict) or not isinstance(teacher_aux, dict):
            return None
        if "layer_cls" not in student_aux or "layer_cls" not in teacher_aux:
            return None

        s = student_aux["layer_cls"]
        t = teacher_aux["layer_cls"]
        if not (torch.is_tensor(s) and torch.is_tensor(t)):
            return None

        # Align number of layers if needed
        if s.ndim != 3 or t.ndim != 3:
            return None
        if s.shape[1] != t.shape[1]:
            L = min(s.shape[1], t.shape[1])
            s = s[:, :L, :]
            t = t[:, :L, :]

        s = F.normalize(s, dim=-1)
        t = F.normalize(t, dim=-1)
        return F.mse_loss(s, t, reduction="mean")

    def compute_shallow_kd_loss(self, student_aux, teacher_aux):
        """Deep-to-shallow logits distillation.

        Teacher provides deep logits; student shallow logits are trained to match via KL.
        Requires aux keys: student_aux['shallow_logits'], teacher_aux['deep_logits'].
        """
        if not isinstance(student_aux, dict) or not isinstance(teacher_aux, dict):
            return None

        t = teacher_aux.get("enhance_logits", None)
        # t = teacher_aux.get("deep_logits", None)
        if t is None or not torch.is_tensor(t):
            return None

        # Prefer multi-depth shallow logits (B, K, C).
        s_multi = student_aux.get("shallow_logits_multi", None)
        # if torch.is_tensor(s_multi) and s_multi.ndim == 3:
        k = int(s_multi.shape[1])
        if k <= 0:
            return None
        losses = []
        for i in range(k):
            losses.append(self.compute_kd_loss(s_multi[:, i, :], t.detach()))

        # print('shallow cls layer')
        return torch.stack(losses).mean()

        # Fallback to single shallow logits.
        # s = student_aux.get("shallow_logits", None)
        # if s is None or not torch.is_tensor(s):
        #     return None
        # return self.compute_kd_loss(s, t.detach())

    def compute_shallow_ce_loss(self, student_aux, y):
        """Supervise per-layer shallow logits with the original CE loss.

        This makes each shallow classifier learn discriminative / semantic features,
        instead of relying only on distillation.
        """
        if not isinstance(student_aux, dict):
            return None

        s_multi = student_aux.get("shallow_logits_multi", None)
        if not (torch.is_tensor(s_multi) and s_multi.ndim == 3):
            return None

        y_target = y.long()

        k = int(s_multi.shape[1])
        if k <= 0:
            return None

        losses = [self.ce_loss_func(s_multi[:, i, :], y_target) for i in range(k)]
        return torch.stack(losses).mean()

    @staticmethod
    def _pairwise_cosine(x: torch.Tensor, y: torch.Tensor):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return x @ y.t()

    def _conle_contrastive(self, x: torch.Tensor, y: torch.Tensor, tau: float):
        """PyTorch version of ConLE `_con`.

        x, y: (B, D) matched pairs (i-th is positive).
        """
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError(f"ConLE expects x,y as (B,D) with same B; got {x.shape} and {y.shape}")

        B = x.shape[0]
        if B <= 1:
            return x.new_zeros(())

        tau = max(float(tau), 1e-6)

        sim_xy = self._pairwise_cosine(x, y)
        sim_xx = self._pairwise_cosine(x, x)

        C = torch.exp(sim_xy / tau)
        # Match the TF code: exp((cos(x,x) - I)/tau). Diagonal becomes exp(0)=1.
        CX = torch.exp((sim_xx - torch.eye(B, device=x.device, dtype=sim_xx.dtype)) / tau) #

        numer = torch.diag(C)
        denom = CX.sum(dim=1) + C.sum(dim=1) - numer
        loss = -torch.log((numer + 1e-12) / (denom + 1e-12))
        return loss.mean()

    def compute_conle_loss(self, image_features, text_features, y, aux=None):
        """Contrastive Label Enhancement (ConLE) loss.

        We reuse existing representations:
        - Z: image_features (B,D)
        - Q: per-sample GT label prototype from text_features (B,D)
        - D: predicted label distribution from logits (softmax)
        - L: target label distribution (one-hot)
        """
        if image_features is None or text_features is None:
            return {}
        if aux is None:
            aux = {}

        y_idx = y.long() if y.dtype != torch.long else y
        Cn = int(text_features.shape[0])
        if Cn <= 0:
            return {}
        y_idx = torch.remainder(y_idx, Cn)

        Z = image_features
        Q = text_features[y_idx]

        con = self._conle_contrastive(Z, Q, self.conle_tau) + self._conle_contrastive(Q, Z, self.conle_tau)

        logits_for_D = aux.get("enhance_logits", None)

        if logits_for_D is None:
            return {"conle_con_loss": con, "conle_loss": con}

        D = F.softmax(logits_for_D, dim=-1)
        L = F.one_hot(y_idx, num_classes=Cn).to(dtype=D.dtype)

        # dis = torch.sum((L - D) ** 2)
        # dis = self.ce_loss_func(D, y_idx)

        # max_neg = torch.max(D * (1 - L), dim=1).values
        # min_pos = torch.min(D * L + (1 - L), dim=1).values
        # thr = torch.mean(torch.clamp(max_neg - min_pos + self.conle_threshold, min=0.0))

        B, C = D.shape
        idx = torch.arange(B, device=D.device)
        pos = D[idx, y_idx]  # [B]
        # 计算等级距离 |k - y|
        levels = torch.arange(C, device=D.device).unsqueeze(0)  # [1, C]
        dist = torch.abs(levels - y_idx.unsqueeze(1))            # [B, C]

        # ranking hinge
        eps = 1e-8
        margin_term = F.relu(D - pos.unsqueeze(1) + eps)
        # 只对负类生效
        loss_mat = dist * margin_term * (1 - L)
        thr = loss_mat.sum(dim=1).mean()

        total = con + self.conle_beta * thr #+ self.conle_alpha * dis
        return {
            "conle_con_loss": con,
            "conle_thr_loss": thr,
            "conle_loss": total,
        }

    @torch.no_grad()
    def update_ema_model(self, source_model: nn.Module, target_model: nn.Module, decay: float):
        """EMA update: target = decay * target + (1-decay) * source.

        Also copies buffers from source to target.
        """
        decay = float(decay)
        source_params = dict(source_model.named_parameters())
        target_params = dict(target_model.named_parameters())
        for name, src in source_params.items():
            tgt = target_params.get(name, None)
            if tgt is None:
                continue
            tgt.data.mul_(decay).add_(src.data, alpha=1.0 - decay)

        source_buffers = dict(source_model.named_buffers())
        target_buffers = dict(target_model.named_buffers())
        for name, srcb in source_buffers.items():
            tgtb = target_buffers.get(name, None)
            if tgtb is None:
                continue
            tgtb.data.copy_(srcb.data)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None, **kwargs):
        # Let Lightning handle the actual optimizer step.
        if optimizer_closure is not None:
            super().optimizer_step(
                epoch,
                batch_idx,
                optimizer,
                optimizer_idx,
                optimizer_closure=optimizer_closure,
                **kwargs,
            )
        else:
            super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, **kwargs)

        # EMA-only: update teacher after the (single) student optimizer step.
        if optimizer_idx == 0:
            try:
                self.update_ema_model(self.module, self.teacher, decay=self.ema_decay)
            except Exception as e:
                logger.warning(f"teacher EMA(student) update skipped: {e}")

    def run_step(self, batch, batch_idx, mode="train"):
        x, y = batch
        res = self.module(x)
        logits, _, image_features, text_features, _, aux = self._unpack_model_output(res)
        losses = self.compute_losses(logits, y)

        if mode == "train" and float(self.loss_weights.get("shallow_ce_loss", 0.0)) > 0:
            shallow_ce = self.compute_shallow_ce_loss(aux, y)
            if shallow_ce is not None:
                losses["shallow_ce_loss"] = shallow_ce

        if mode == "train" and float(self.loss_weights.get("kd_loss", 0.0)) > 0:
            with torch.no_grad():
                teacher_res = self.teacher(x)
            teacher_logits, _, _, _, _, teacher_aux = self._unpack_model_output(teacher_res)
            enhance_logits = teacher_aux["enhance_logits"]
            losses["kd_loss"] = (
                self.compute_kd_loss(logits, teacher_logits.detach())
                + self.compute_kd_loss(logits, enhance_logits.detach())
            )

            if float(self.loss_weights.get("shallow_kd_loss", 0.0)) > 0:
                shallow_kd = self.compute_shallow_kd_loss(aux, teacher_aux)
                if shallow_kd is not None:
                    losses["shallow_kd_loss"] = shallow_kd

            if float(self.loss_weights.get("feat_kd_loss", 0.0)) > 0:
                feat_kd = self.compute_feat_kd_loss(aux, teacher_aux)
                if feat_kd is not None:
                    losses["feat_kd_loss"] = feat_kd

        if mode == "train" and float(self.loss_weights.get("conle_loss", 0.0)) > 0:
            aux_local = dict(aux or {})
            aux_local["logits"] = logits
            losses.update(self.compute_conle_loss(image_features, text_features, y, aux_local))

        loss = self.aggregate_loss(losses)
        metrics_exp = self.compute_per_example_metrics(logits, y, "exp")
        metrics_max = self.compute_per_example_metrics(logits, y, "max")
        outputs = {"loss": loss, **losses, **metrics_exp, **metrics_max}
        if mode in {"val", "test"}:
            outputs.update(self.compute_logits_unimodality(logits))
        if mode == "val":
            return outputs, text_features
        return outputs

    def on_save_checkpoint(self, checkpoint):
        checkpoint["extra_value"] = getattr(self, "extra_value", None)
        # print("extra_value added to checkpoint")

    def _split_batch(self, batch):
        x, y = batch
        return x, y, None

    def _record_track10_enhance_logits(self, outputs: dict, batch_idx: int):
        if not isinstance(outputs, dict) or "track10_enhance_logits" not in outputs:
            return

        try:
            logits = outputs["track10_enhance_logits"].detach().float().cpu()
            probs = torch.softmax(logits, dim=-1)

            indices = outputs.get("track10_indices", None)
            labels = outputs.get("track10_labels", None)
            if not (torch.is_tensor(indices) and torch.is_tensor(labels)):
                return
            indices = indices.detach().cpu().tolist()
            labels = labels.detach().cpu().tolist()
            image_paths = outputs.get("track10_image_paths", None)
            if not isinstance(image_paths, list):
                image_paths = [""] * len(indices)

            step = int(self.global_step)
            epoch = int(self.current_epoch)

            for i in range(logits.shape[0]):
                sample_idx = int(indices[i])
                target = int(labels[i])
                sample_path = str(image_paths[i]) if i < len(image_paths) else ""
                # Per-sample distribution summary to track shift trend.
                entropy = float(-(probs[i] * (probs[i].clamp(min=1e-12).log())).sum().item())
                self._track10_history_rows.append(
                    {
                        "global_step": step,
                        "epoch": epoch,
                        "batch_idx": int(batch_idx),
                        "sample_index": sample_idx,
                        "image_path": sample_path,
                        "target": target,
                        "class_index": -1,
                        "logit": float("nan"),
                        "prob": float("nan"),
                        "entropy": entropy,
                    }
                )

                for c in range(logits.shape[1]):
                    self._track10_history_rows.append(
                        {
                            "global_step": step,
                            "epoch": epoch,
                            "batch_idx": int(batch_idx),
                            "sample_index": sample_idx,
                            "image_path": sample_path,
                            "target": target,
                            "class_index": int(c),
                            "logit": float(logits[i, c].item()),
                            "prob": float(probs[i, c].item()),
                            "entropy": float("nan"),
                        }
                    )
        except Exception as e:
            self._custom_logger.warning(f"record track10 enhance_logits skipped: {e}")

    def _dump_track10_history(self):
        if len(self._track10_history_rows) == 0:
            return

        out_dir = self.output_dir / "track10_enhance_logits"
        out_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self._track10_history_rows)
        csv_path = out_dir / "track10_enhance_logits.csv"
        df.to_csv(csv_path, index=False)

        per_class_df = df[df["class_index"] >= 0].copy()
        summary_df = df[df["class_index"] < 0].copy()

        # Trend plot 1: average class-probability distribution over steps (heatmap).
        if len(per_class_df) > 0:
            agg = (
                per_class_df.groupby(["global_step", "class_index"], as_index=False)["prob"]
                .mean()
            )
            piv = agg.pivot(index="class_index", columns="global_step", values="prob").sort_index()
            plt.figure(figsize=(12, 4))
            sns.heatmap(piv, cmap="viridis")
            plt.xlabel("Global step")
            plt.ylabel("Class index")
            plt.title("Track10 mean probability distribution over steps")
            plt.tight_layout()
            plt.savefig(out_dir / "track10_prob_distribution_heatmap.png")
            plt.close()

            # Trend plot 2: true-class probability curve for each tracked sample.
            true_prob_df = per_class_df[per_class_df["class_index"] == per_class_df["target"]].copy()
            if len(true_prob_df) > 0:
                plt.figure(figsize=(12, 5))
                for sid in sorted(true_prob_df["sample_index"].unique().tolist()):
                    sdf = true_prob_df[true_prob_df["sample_index"] == sid].sort_values("global_step")
                    plt.plot(sdf["global_step"], sdf["prob"], label=f"sample_{sid}", linewidth=1.5)
                plt.xlabel("Global step")
                plt.ylabel("True-class probability")
                plt.title("Track10 true-class probability trend")
                plt.legend(ncol=2, fontsize=8)
                plt.tight_layout()
                plt.savefig(out_dir / "track10_true_class_prob_trend.png")
                plt.close()

        # Optional summary trend: entropy mean over steps.
        if len(summary_df) > 0:
            e = summary_df.groupby("global_step", as_index=False)["entropy"].mean()
            plt.figure(figsize=(10, 4))
            plt.plot(e["global_step"], e["entropy"], linewidth=1.8)
            plt.xlabel("Global step")
            plt.ylabel("Mean entropy")
            plt.title("Track10 distribution entropy trend")
            plt.tight_layout()
            plt.savefig(out_dir / "track10_entropy_trend.png")
            plt.close()

    def training_step(self, batch, batch_idx, optimizer_idx: int = 0):
        outputs = self.run_step(batch, batch_idx, "train")
        # Record tracking samples every 100 global steps to reduce overhead.
        if int(self.global_step) % 100 == 0:
            fixed_track = self._collect_fixed_tracking_enhance_logits()
            if isinstance(outputs, dict) and fixed_track is not None:
                outputs["track10_indices"] = fixed_track["sample_indices"]
                outputs["track10_labels"] = fixed_track["labels"]
                outputs["track10_enhance_logits"] = fixed_track["enhance_logits"]
                outputs["track10_image_paths"] = fixed_track["image_paths"]
            self._record_track10_enhance_logits(outputs, batch_idx)
        self.logging(outputs, "train", on_step=True, on_epoch=True)
        return outputs

    def on_train_end(self) -> None:
        self._dump_track10_history()

    def validation_step(self, batch, batch_idx):
        res = self.run_step(batch, batch_idx, "val")
        if isinstance(res, tuple):
            outputs, text_features = res
            self.extra_value = text_features
        else:
            outputs = res

        # # Make sure val_* metrics exist for ModelCheckpoint monitor keys.
        # self.logging(outputs, "val", on_step=True, on_epoch=True)

        # # step级展示：当前batch单峰率(均值)
        # if isinstance(outputs, dict) and ("unimodal_ratio_metric" in outputs):
        #     self.log(
        #         "val_unimodal_ratio_metric",
        #         outputs["unimodal_ratio_metric"].float().mean(),
        #         on_step=True,
        #         on_epoch=False,
        #         prog_bar=True,
        #         logger=True,
        #     )

        return outputs

    # def test_step(self, batch, batch_idx):
    #     outputs = self.run_step(batch, batch_idx, "test")

    #     return outputs

    def test_step(self, batch, batch_idx):
        x, y = batch
        res = self.module(x)
        logits, _, image_features, text_features, patch_features, _ = self._unpack_model_output(res)
        losses = self.compute_losses(logits, y)
        loss = self.aggregate_loss(losses)
        metrics_exp = self.compute_per_example_metrics(logits, y, "exp")
        metrics_max = self.compute_per_example_metrics(logits, y, "max")
        unimodal_metrics = self.compute_logits_unimodality(logits)
        outputs = {"loss": loss, 'image_features': image_features, 'text_features': text_features,
                   'patch_features': patch_features, 'ground_truth':y, 'logits':logits ,
                   **losses, **metrics_exp, **metrics_max, **unimodal_metrics}

        return outputs

    # Epoch Evals
    def eval_epoch_end(self, outputs, run_type, extra_stats=None):
        """_summary_

        Args:
            outputs (_type_): _description_
            run_type (_type_): _description_
            moniter_key: "{val/test}_epoch_{mae/acc}_{exp/max}_metric"
        """
        stats = defaultdict(list)
        for _outputs in outputs:
            for k, v in _outputs.items():
                if self._valid_key(k):
                    stats[k].append(v)
        for k, _stats in stats.items():
            try:
                stats[k] = torch.cat(_stats).mean().item()
            except RuntimeError:
                stats[k] = torch.stack(_stats).mean().item()
            self.log(f"{run_type}_{k}", stats[k], on_step=False, on_epoch=True, prog_bar=False, logger=True)

        if extra_stats:
            for k, v in extra_stats.items():
                stats[k] = float(v)
                self.log(f"{run_type}_{k}", float(v), on_step=False, on_epoch=True, prog_bar=False, logger=True)

        stats["epoch"] = self.current_epoch
        stats["output_dir"] = str(self.output_dir)
        stats["ckpt_path"] = str(self.ckpt_path)
        with open(str(self.output_dir / f"{run_type}_stats.json"), "a") as f:
            f.write(json.dumps(stats) + "\n")

    def validation_epoch_end(self, outputs) -> None:
        self.eval_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs) -> None:
        self.eval_epoch_end(outputs, "test")

    def on_train_epoch_start(self) -> None:
        opts = self.optimizers()

        def _summarize_optimizer(opt):
            param_group_lrs = {}
            for i, pg in enumerate(getattr(opt, "param_groups", [])):
                name = pg.get("name", f"group_{i}")
                params = pg.get("params", [])
                try:
                    n_params = len(params)
                except TypeError:
                    n_params = len(list(params))
                param_group_lrs[name] = (pg.get("lr", None), n_params)
            return param_group_lrs

        if isinstance(opts, (list, tuple)):
            for i, opt in enumerate(opts):
                summary = _summarize_optimizer(opt)
                # logger.info(
                #     f"check optimizer[{i}] `param_groups` lr @ epoch {self.current_epoch}: {summary}"
                # )
        else:
            summary = _summarize_optimizer(opts)
            # logger.info(f"check optimizer `param_groups` lr @ epoch {self.current_epoch}: {summary}")

    def on_fit_start(self) -> None:
        pl.seed_everything(self.seed, workers=True)

    # Logging Utils
    loggings_suffix = {"metric", "loss", "acc", "mae"}

    def _valid_key(self, key: str):
        for suffix in self.loggings_suffix:
            if key.endswith(suffix):
                return True
        else:
            return False

    def logging(self, outputs: dict, run_type: str, on_step=True, on_epoch=True):
        for k, v in outputs.items():
            if self._valid_key(k):
                self.log(f"{run_type}_{k}", v.mean(), on_step=on_step, on_epoch=on_epoch, prog_bar=False, logger=True)

    # Loss & Metrics
    def compute_losses(self, logits, y):
        zero = logits.new_zeros(1)
        return {
            "ce_loss": self.ce_loss_func(logits, y),
            "kl_loss": zero,
            "reg_loss": zero,
        }

    def compute_kd_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor):
        """KL-based distillation loss on logits.

        Uses standard temperature-scaled KL:
            KL( softmax(teacher/T) || softmax(student/T) ) * T^2
        """
        T = max(self.kd_temp, 1e-6)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        # batchmean is preferable, but we keep existing KLDivLoss(reduction='sum')
        # and normalize by batch size for stability.
        kl = self.kl_loss_func(student_log_probs, teacher_probs)
        kl = kl / max(int(student_logits.shape[0]), 1)
        return kl * (T * T)

    def compute_per_example_metrics(self, logits, y, gather_type="exp"):

        probs = F.softmax(logits, -1)
        dtype = logits.dtype

        if gather_type == "exp":

            rank_output_value_array = self.rank_output_value_array.type(dtype)
            predict_y_label = torch.sum(probs * rank_output_value_array, dim=-1)

        elif gather_type == "max":
            predict_y_label = torch.argmax(probs, dim=-1).type(dtype)

        else:
            raise ValueError(f"Invalid gather_type: {gather_type}")

        y = y.type(dtype)
        mae = torch.abs(predict_y_label - y)
        acc = (torch.round(predict_y_label) == y).type(logits.dtype)
        acc_adj = (torch.abs(torch.round(predict_y_label) - y) <= 1).type(logits.dtype)
        return {
            f"mae_{gather_type}_metric": mae,
            f"acc_{gather_type}_metric": acc,
            "predict_y": predict_y_label,
            f"acc_{gather_type}_adj_metric": acc_adj,
        }


    def compute_logits_unimodality(self, logits, tol: float = 1e-6):
        """Check if each probability vector has a single peak (unimodal).

        Args:
            logits (torch.Tensor): Raw model outputs of shape (N, C).
            tol (float): Numerical tolerance when checking monotonic segments.


        Returns:
            dict: contains `unimodal_ratio_metric` (tensor, shape (N,), values in {0,1}).
        """

        probs = F.softmax(logits, dim=-1)
        diffs = probs[:, 1:] - probs[:, :-1]  # (N, C-1)
        peaks = probs.argmax(dim=-1)  # (N,)

        idx = torch.arange(diffs.shape[1], device=logits.device).unsqueeze(0)  # (1, C-1)
        peak_idx = peaks.unsqueeze(1)  # (N, 1)

        left_viol = (diffs < -tol) & (idx < peak_idx)
        right_viol = (diffs > tol) & (idx >= peak_idx)
        unimodal_mask = ~(left_viol.any(dim=1) | right_viol.any(dim=1))

        # 返回逐样本0/1，epoch_end用cat+mean得到全eval-set加权单峰率
        return {"unimodal_ratio_metric": unimodal_mask.float()}


    # Optimizer & Scheduler
    def configure_optimizers(self):
        return self.build_optmizer_and_scheduler(**self._optimizer_and_scheduler_cfg)

    def build_optmizer_and_scheduler(
        self,
        param_dict_cfg=None,
        optimizer_cfg=None,
        lr_scheduler_cfg=None,
    ):
        param_dict_ls = self.build_param_dict(**param_dict_cfg)

        optim = build_optimizer(
            model=param_dict_ls,
            **optimizer_cfg,
        )
        sched = build_lr_scheduler(optimizer=optim, **lr_scheduler_cfg)
        return [optim], [sched]

    # Model IO
    def load_weights(
        self,
        init_model_weights=None,
        init_prompt_learner_weights=None,
        init_image_encoder_weights=None,
        init_image_adapter_weights=None,
        init_text_encoder_weights=None,
        init_regressor_weights=None,

    ):
        if init_model_weights is not None:
            self._custom_logger.info("init_model_weights")
            load_pretrained_weights(self.module, init_model_weights)
            return

        if init_prompt_learner_weights is not None:
            self._custom_logger.info("init_prompt_learner_weights")
            load_pretrained_weights(self.module.prompt_learner, init_prompt_learner_weights)
        if init_image_encoder_weights is not None:
            self._custom_logger.info("init_image_encoder_weights")
            load_pretrained_weights(self.module.image_encoder, init_image_encoder_weights)
        if init_image_adapter_weights is not None:
            self._custom_logger.info("init_image_adapter_weights")
            load_pretrained_weights(self.module.image_adapter, init_image_adapter_weights)
        if init_text_encoder_weights is not None:
            self._custom_logger.info("init_prompt_learner_weights")
            load_pretrained_weights(self.module.text_encoder, init_text_encoder_weights)
        if init_regressor_weights is not None:
            self._custom_logger.info("init_regressor_weights")
            load_pretrained_weights(self.module.regressor, init_regressor_weights)
        return

    def build_param_dict(
        self,
        lr_prompt_learner_context,
        lr_prompt_learner_ranks,
        lr_image_encoder,
        lr_image_adapter,
        lr_text_encoder,
        lr_logit_scale,
        staged_lr_image_encoder,
        lr_regressor,
        model=None
    ):
        model = self.module if model is None else model

        def _as_float(v, default):
            if v is None:
                return float(default)
            return float(v)

        # When a module lr is not explicitly configured, default to lr_image_adapter.
        base_lr = float(lr_image_adapter)
        lr_prompt_learner_context = _as_float(lr_prompt_learner_context, base_lr)
        lr_prompt_learner_ranks = _as_float(lr_prompt_learner_ranks, base_lr)
        lr_image_encoder = _as_float(lr_image_encoder, base_lr)
        lr_text_encoder = _as_float(lr_text_encoder, base_lr)
        lr_logit_scale = _as_float(lr_logit_scale, base_lr)
        lr_regressor = _as_float(lr_regressor, base_lr)

        param_dict_ls = []
        assigned_param_ids = set()

        def _iter_params(obj):
            if obj is None:
                return []
            if isinstance(obj, torch.nn.Parameter):
                return [obj]
            if isinstance(obj, nn.Module):
                return list(obj.parameters())
            if isinstance(obj, (list, tuple)):
                out = []
                for it in obj:
                    out.extend(_iter_params(it))
                return out
            return []

        def _mark_assigned(params):
            for p in params:
                assigned_param_ids.add(id(p))

        def _add_group(name: str, params, lr_value: float):
            ps = [p for p in _iter_params(params) if isinstance(p, torch.nn.Parameter) and p.requires_grad]
            if len(ps) == 0:
                return
            _mark_assigned(ps)
            param_dict_ls.append(
                {
                    "params": ps,
                    "lr": float(lr_value),
                    "init_lr": float(lr_value),
                    "name": name,
                }
            )

        def _freeze_if_zero(lr_value: float, target, desc: str):
            if lr_value == 0.0 and target is not None:
                self._custom_logger.info(f"freeze_param({desc}) because lr==0")
                freeze_param(target)
                return True
            return False

        # 1) Freeze only when corresponding lr == 0.
        if getattr(model, "prompt_learner", None) is not None:
            _freeze_if_zero(lr_prompt_learner_context, getattr(model.prompt_learner, "context_embeds", None), "prompt_learner.context_embeds")
            _freeze_if_zero(lr_prompt_learner_ranks, getattr(model.prompt_learner, "rank_embeds", None), "prompt_learner.rank_embeds")
        _freeze_if_zero(lr_image_encoder, getattr(model, "image_encoder", None), "image_encoder")
        _freeze_if_zero(base_lr, getattr(model, "image_adapter", None), "image_adapter")
        _freeze_if_zero(lr_text_encoder, getattr(model, "text_encoder", None), "text_encoder")
        _freeze_if_zero(lr_logit_scale, getattr(model, "logit_scale", None), "logit_scale")
        _freeze_if_zero(lr_regressor, getattr(model, "regressor", None), "regressor")

        # 2) Dedicated lr groups (only if lr != 0). Any remaining params fall back to default lr.
        if getattr(model, "prompt_learner", None) is not None:
            if lr_prompt_learner_context != 0.0 and hasattr(model.prompt_learner, "context_embeds"):
                _add_group("lr_prompt_learner_context", model.prompt_learner.context_embeds, lr_prompt_learner_context)
            if lr_prompt_learner_ranks != 0.0 and hasattr(model.prompt_learner, "rank_embeds"):
                _add_group("lr_prompt_learner_ranks", model.prompt_learner.rank_embeds, lr_prompt_learner_ranks)

        if getattr(model, "image_encoder", None) is not None and lr_image_encoder != 0.0:
            if staged_lr_image_encoder is not None:
                self._custom_logger.info("staged_lr_image_encoder activated")
                image_encoder_param_groups = build_staged_lr_param_groups(
                    model=model.image_encoder,
                    lr=lr_image_encoder,
                    **staged_lr_image_encoder,
                )
                # mark assigned to avoid duplicating into default group
                for g in image_encoder_param_groups:
                    ps = [p for p in g.get("params", []) if isinstance(p, torch.nn.Parameter) and p.requires_grad]
                    _mark_assigned(ps)
                param_dict_ls.extend(image_encoder_param_groups)
            else:
                _add_group("image_encoder", model.image_encoder, lr_image_encoder)

        if getattr(model, "image_adapter", None) is not None and base_lr != 0.0:
            _add_group("image_adapter", model.image_adapter, base_lr)

        if getattr(model, "text_encoder", None) is not None and lr_text_encoder != 0.0:
            _add_group("text_encoder", model.text_encoder, lr_text_encoder)

        if getattr(model, "logit_scale", None) is not None and lr_logit_scale != 0.0:
            _add_group("logit_scale", model.logit_scale, lr_logit_scale)

        if getattr(model, "regressor", None) is not None and lr_regressor != 0.0:
            _add_group("regressor", model.regressor, lr_regressor)

        # 3) Default group: all remaining trainable params (including newly added modules/params).
        default_params = [
            p
            for p in model.parameters()
            if p.requires_grad and (id(p) not in assigned_param_ids)
        ]
        if len(default_params) > 0 and base_lr != 0.0:
            param_dict_ls.append(
                {
                    "params": default_params,
                    "lr": base_lr,
                    "init_lr": base_lr,
                    "name": "default",
                }
            )

        # 4) Print trainable params that are actually passed to the optimizer.
        # NOTE: This function is called during optimizer construction, so printing here
        # won't spam every step/epoch.
        try:
            name_by_id = {id(p): n for n, p in model.named_parameters()}

            group_summaries = []
            included_ids = set()
            for i, g in enumerate(param_dict_ls):
                gname = g.get("name", f"group_{i}")
                glr = g.get("lr", None)
                ps = g.get("params", [])
                ps = [p for p in ps if isinstance(p, torch.nn.Parameter) and p.requires_grad]
                for p in ps:
                    included_ids.add(id(p))

                n_tensors = len(ps)
                n_elems = int(sum(int(p.numel()) for p in ps))
                group_summaries.append((gname, glr, n_tensors, n_elems, ps))

            total_tensors = sum(s[2] for s in group_summaries)
            total_elems = sum(s[3] for s in group_summaries)
            self._custom_logger.info(
                f"[build_param_dict] optimizer params: groups={len(group_summaries)}, tensors={total_tensors}, elems={total_elems}"
            )

            # Per-group detail + names (truncate if too many)
            max_names_to_print = 20
            for (gname, glr, n_tensors, n_elems, ps) in group_summaries:
                self._custom_logger.info(
                    f"[build_param_dict] group={gname} lr={glr} tensors={n_tensors} elems={n_elems}"
                )
                names = [name_by_id.get(id(p), f"<unnamed:{id(p)}>" ) for p in ps]
                if len(names) <= max_names_to_print:
                    self._custom_logger.info(f"[build_param_dict] group={gname} params={names}")
                else:
                    head = names[:max_names_to_print]
                    # self._custom_logger.info(
                    #     f"[build_param_dict] group={gname} params(head {max_names_to_print}/{len(names)})={head}"
                    # )

            # Also print unassigned-but-trainable (should be empty unless base_lr==0)
            remaining_trainable = [
                n for n, p in model.named_parameters()
                if p.requires_grad and (id(p) not in included_ids)
            ]
            if len(remaining_trainable) > 0:
                self._custom_logger.info(
                    f"[build_param_dict] WARNING: trainable params not in optimizer groups: {remaining_trainable[:max_names_to_print]}"
                )
        except Exception as e:
            self._custom_logger.info(f"[build_param_dict] param print skipped: {e}")

        return param_dict_ls
