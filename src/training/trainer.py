"""Supervised trainer for the VLA action head."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models.lora_utils import disable_peft_bitsandbytes_dispatch
from src.models.vla_model import VLAModel
from src.training.losses import action_loss


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainerConfig:
    """Training settings for supervised action-head optimization."""

    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    pin_memory: bool = True
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 5
    early_stopping_patience: int = 10
    seed: int = 42
    use_amp: bool = True
    use_wandb: bool = False
    wandb_project: str = "vla-av"
    max_train_batches: Optional[int] = None
    max_val_batches: Optional[int] = None
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Tuple[str, ...] = ("q_proj", "v_proj")
    gradient_checkpointing: bool = False
    domain_loss_weight: float = 0.0
    domain_pair_batch_size: Optional[int] = None
    action_weights: Tuple[float, float, float] = (2.0, 1.0, 1.0)
    show_progress: bool = True


class VLATrainer:
    """Train the action head, optionally with LoRA adapters in the VLM backbone."""

    def __init__(
        self,
        model: VLAModel,
        train_dataset: Dataset,
        val_dataset: Dataset,
        config: Optional[TrainerConfig] = None,
        *,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or TrainerConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        if self.config.use_lora:
            self._prepare_lora_backbone()
        else:
            self.model.freeze_backbone()
            self.model.backbone.eval()

        self.model.action_head.to(self.device)

        trainable_parameters = self._trainable_parameters()
        self.optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.use_amp = self.config.use_amp and self.device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)
        self.wandb_run = self._init_wandb()

    def fit(self) -> Dict[str, float]:
        """Run train/val epochs with checkpointing and early stopping."""

        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        last_train_loss = float("inf")
        last_val_loss = float("inf")

        for epoch in range(1, self.config.epochs + 1):
            last_train_loss = self.train_epoch()
            last_val_loss = self.validate()

            print(
                f"Epoch {epoch}/{self.config.epochs} | "
                f"train_loss={last_train_loss:.3f} | val_loss={last_val_loss:.3f}",
                flush=True,
            )
            self._log_metrics(epoch, last_train_loss, last_val_loss)

            if last_val_loss < best_val_loss:
                best_val_loss = last_val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                self.save_checkpoint("best_model.pt", epoch, last_train_loss, last_val_loss)
            else:
                epochs_without_improvement += 1

            if epoch % self.config.checkpoint_every == 0:
                self.save_checkpoint(
                    f"checkpoint_epoch_{epoch:03d}.pt",
                    epoch,
                    last_train_loss,
                    last_val_loss,
                )

            if epochs_without_improvement >= self.config.early_stopping_patience:
                LOGGER.info(
                    "Early stopping at epoch %s; best val_loss %.6f at epoch %s.",
                    epoch,
                    best_val_loss,
                    best_epoch,
                )
                break

        if self.wandb_run is not None:
            self.wandb_run.finish()

        return {
            "train_loss": last_train_loss,
            "val_loss": last_val_loss,
            "best_val_loss": best_val_loss,
            "best_epoch": float(best_epoch),
        }

    def train_epoch(self) -> float:
        """Train the action head for one epoch."""

        self.model.action_head.train()
        if self.config.use_lora:
            self.model.backbone.train()
        else:
            self.model.backbone.eval()
        loader = self._make_loader(self.train_dataset, shuffle=True)
        return self._run_batches(
            loader,
            train=True,
            max_batches=self.config.max_train_batches,
            description="train",
        )

    @torch.no_grad()
    def validate(self) -> float:
        """Evaluate validation loss without optimizer updates."""

        self.model.action_head.eval()
        self.model.backbone.eval()
        loader = self._make_loader(self.val_dataset, shuffle=False)
        return self._run_batches(
            loader,
            train=False,
            max_batches=self.config.max_val_batches,
            description="val",
        )

    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        train_loss: float,
        val_loss: float,
    ) -> Path:
        """Save action-head weights and training metadata."""

        path = self.checkpoint_dir / filename
        payload = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "model_type": getattr(self.model, "model_type", "vla"),
            "action_head_type": getattr(self.model, "action_head_type", "mlp"),
            "action_head_config": asdict(self.model.action_head.config)
            if hasattr(self.model.action_head, "config")
            else {},
            "action_head_state_dict": self.model.action_head.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "trainer_config": asdict(self.config),
        }
        if self.config.use_lora:
            payload["backbone_lora_state_dict"] = self._lora_state_dict()
            payload["lora_config"] = {
                "r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "lora_dropout": self.config.lora_dropout,
                "target_modules": list(self.config.lora_target_modules),
                "task_type": "FEATURE_EXTRACTION",
            }

        torch.save(payload, path)
        return path

    def _run_batches(
        self,
        loader: DataLoader,
        *,
        train: bool,
        max_batches: Optional[int],
        description: str,
    ) -> float:
        total_loss = 0.0
        total_examples = 0
        total_batches = len(loader)
        if max_batches is not None:
            total_batches = min(total_batches, max_batches)
        iterator = enumerate(loader)
        if self.config.show_progress:
            iterator = tqdm(
                iterator,
                total=total_batches,
                desc=description,
                unit="batch",
                leave=False,
            )

        for batch_idx, (images, instructions, targets) in iterator:
            if max_batches is not None and batch_idx >= max_batches:
                break

            if train:
                targets = targets.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                with autocast("cuda", enabled=self.use_amp):
                    supervised_loss = self._supervised_action_loss(
                        images,
                        instructions,
                        targets,
                        train=True,
                    )
                    domain_loss = self._domain_adaptation_loss(
                        batch_size=int(targets.shape[0]),
                        train=True,
                    )
                    loss = supervised_loss + domain_loss
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                targets = targets.to(self.device, non_blocking=True)
                with torch.no_grad():
                    with autocast("cuda", enabled=self.use_amp):
                        loss = self._supervised_action_loss(
                            images,
                            instructions,
                            targets,
                            train=False,
                        )

            batch_size = int(targets.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_examples += batch_size
            if self.config.show_progress and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

        if total_examples == 0:
            raise RuntimeError("No batches were processed during training/validation.")
        return total_loss / total_examples

    def _encode_batch(self, images: torch.Tensor, instructions: Iterable[str]) -> torch.Tensor:
        image_list = [image for image in images]
        instruction_list = list(instructions)
        embeddings = self.model.backbone(image_list, instruction_list)
        return embeddings.to(self.device, non_blocking=True, dtype=torch.float32)

    def _forward_batch(
        self,
        images: torch.Tensor,
        instructions: Iterable[str],
        *,
        train: bool,
    ) -> torch.Tensor:
        if self.config.use_lora and train:
            image_list = [image for image in images]
            instruction_list = list(instructions)
            return self.model(image_list, instruction_list)

        with torch.no_grad():
            embeddings = self._encode_batch(images, instructions)
        return self.model.action_head(embeddings)

    def _supervised_action_loss(
        self,
        images: torch.Tensor,
        instructions: Iterable[str],
        targets: torch.Tensor,
        *,
        train: bool,
    ) -> torch.Tensor:
        if self._uses_diffusion_head():
            if self.config.use_lora and train:
                image_list = [image for image in images]
                instruction_list = list(instructions)
                embeddings = self.model.backbone(image_list, instruction_list).to(
                    self.device,
                    non_blocking=True,
                    dtype=torch.float32,
                )
            else:
                with torch.no_grad():
                    embeddings = self._encode_batch(images, instructions)
            return self.model.action_head.diffusion_loss(
                embeddings,
                targets,
                weights=self._action_weights_tensor(targets),
            )

        preds = self._forward_batch(images, instructions, train=train)
        return action_loss(
            preds,
            targets,
            weights=self._action_weights_tensor(preds),
        )

    def _domain_adaptation_loss(self, *, batch_size: int, train: bool) -> torch.Tensor:
        """Align real and Cosmos embeddings when MixedDataset can provide pairs."""

        zero = torch.zeros((), device=self.device, dtype=torch.float32)
        if (
            not train
            or self.config.domain_loss_weight <= 0.0
            or not self.config.use_lora
        ):
            return zero

        sampler = getattr(self.train_dataset, "sample_domain_pairs", None)
        if sampler is None:
            return zero

        pair_batch_size = self.config.domain_pair_batch_size or max(1, min(batch_size, self.config.batch_size))
        pairs = sampler(pair_batch_size)
        if pairs is None:
            return zero

        real_images, real_instructions, synthetic_images, synthetic_instructions = pairs
        real_embeddings = self._encode_batch(real_images, real_instructions)
        synthetic_embeddings = self._encode_batch(synthetic_images, synthetic_instructions)
        return F.mse_loss(real_embeddings, synthetic_embeddings) * self.config.domain_loss_weight

    def _make_loader(self, dataset: Dataset, *, shuffle: bool) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(self.config.seed)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory and torch.cuda.is_available(),
            persistent_workers=self.config.num_workers > 0,
            generator=generator,
        )

    def _init_wandb(self) -> Any:
        if not self.config.use_wandb:
            return None

        import wandb

        return wandb.init(
            project=self.config.wandb_project,
            config=asdict(self.config),
        )

    def load_initial_checkpoint(self, path: str | Path) -> None:
        """Initialize LoRA adapters and the action head from a previous run."""

        checkpoint_path = Path(path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")

        action_state = checkpoint.get("action_head_state_dict")
        if action_state is not None:
            try:
                self.model.action_head.load_state_dict(action_state)
            except RuntimeError as exc:
                LOGGER.warning(
                    "Skipping incompatible action head from %s (%s). "
                    "This is expected when initializing GR00T diffusion from an MLP checkpoint.",
                    checkpoint_path,
                    exc.__class__.__name__,
                )
            else:
                LOGGER.info("Initialized action head from %s", checkpoint_path)

        lora_state = checkpoint.get("backbone_lora_state_dict")
        if lora_state is not None and self.config.use_lora:
            try:
                from peft import set_peft_model_state_dict
            except ImportError as exc:  # pragma: no cover - optional dependency.
                raise RuntimeError("PEFT is required to initialize LoRA adapters.") from exc
            set_peft_model_state_dict(self.model.backbone.model, lora_state)
            LOGGER.info("Initialized LoRA adapters from %s", checkpoint_path)

    def _log_metrics(self, epoch: int, train_loss: float, val_loss: float) -> None:
        if self.wandb_run is None:
            return

        self.wandb_run.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
        )

    def _prepare_lora_backbone(self) -> None:
        """Attach LoRA adapters to Qwen2-VL and leave only adapters trainable."""

        disable_peft_bitsandbytes_dispatch()

        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:  # pragma: no cover - optional dependency.
            raise RuntimeError(
                "PEFT is required for LoRA training. Install peft or disable --lora."
            ) from exc

        if not hasattr(self.model.backbone, "model"):
            raise RuntimeError("LoRA training requires a transformer-backed VLM backbone.")

        if self.config.gradient_checkpointing and hasattr(
            self.model.backbone, "enable_gradient_checkpointing"
        ):
            self.model.backbone.enable_gradient_checkpointing()

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=list(self.config.lora_target_modules),
            lora_dropout=self.config.lora_dropout,
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.model.backbone.model = get_peft_model(self.model.backbone.model, lora_config)
        self.model.backbone.train()

        if hasattr(self.model.backbone.model, "print_trainable_parameters"):
            self.model.backbone.model.print_trainable_parameters()

    def _trainable_parameters(self) -> List[torch.nn.Parameter]:
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("No trainable parameters found for VLATrainer.")
        LOGGER.info("Training %s parameter tensors.", len(parameters))
        return parameters

    def _action_weights_tensor(self, reference: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            self.config.action_weights,
            dtype=reference.dtype,
            device=reference.device,
        )

    def _uses_diffusion_head(self) -> bool:
        return callable(getattr(self.model.action_head, "diffusion_loss", None))

    def _lora_state_dict(self) -> Dict[str, torch.Tensor]:
        try:
            from peft import get_peft_model_state_dict
        except ImportError:
            return {}

        if not hasattr(self.model.backbone, "model"):
            return {}
        return get_peft_model_state_dict(self.model.backbone.model)
