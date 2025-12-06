from typing import Callable, Dict, List, Optional, Tuple, Union, Any
import torch
import wandb
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import Trainer, Seq2SeqTrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.trainer import (
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
)
from peft.tuners.lora.layer import Linear as LoraLinear


# Configuration for logging specific layers
INCLUDE_KEYWORDS = ["encoder.block.2", "encoder.block.3", "encoder.block.4"]  # T5
# INCLUDE_KEYWORDS = ["layers.27", "layers.6"]  # Llama example
# INCLUDE_KEYWORDS = ["block.0", "block.4"]  # Alternative example

DO_LOG = True


def get_forward_hook(name):
    """
    Create a forward hook to log input/output statistics.
    
    Args:
        name: Name of the module for logging
        
    Returns:
        Hook function
    """
    def hook(module, input, output):
        wandb.log(
            {
                f"{name}/input_mean": input[0].mean().item(),
                f"{name}/input_std": input[0].std().item(),
                f"{name}/output_mean": output.mean().item(),
                f"{name}/output_std": output.std().item(),
            },
            commit=False,
        )
    return hook


class LogTrainer(Trainer):
    """
    Custom Trainer that logs LoRA parameter statistics during training.
    Tracks gradient norms, parameter norms, and singular value distributions.
    """
    
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: Seq2SeqTrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        log_loss_callback: Optional[Callable[[float, int], None]] = None,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        
        self.log_loss_callback = log_loss_callback
        self.is_peft = "PeftModel" in type(model).__name__
        
        # Get LoRA scaling factor if using PEFT
        if self.is_peft:
            for name, module in model.named_modules():
                if isinstance(module, LoraLinear):
                    self.scaling = module.scaling["default"]
                    break
        
        # Initialize parameter tracking
        self.orig_A = None
        self.orig_B = None
        self.orig_W = None
        self.gradient_accumulation_counter = 0

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Perform a training step with optional logging of parameter statistics.
        
        Args:
            model: The model being trained
            inputs: Input batch
            
        Returns:
            Normalized loss value
        """
        if not DO_LOG:
            return super().training_step(model, inputs)
        
        # Initialize parameter tracking on first step
        if self.is_peft:
            if self.orig_A is None:
                self._initialize_lora_tracking(model)
        else:
            if self.orig_W is None:
                self._initialize_full_weight_tracking(model)
        
        # Standard training step
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        if self.args.n_gpu > 1:
            loss = loss.mean()
        
        self.accelerator.backward(loss)
        
        # Call loss logging callback if provided
        if self.log_loss_callback is not None:
            self.log_loss_callback(loss.item(), self.state.global_step)
        
        # Log parameter statistics after gradient accumulation
        if self._should_log_statistics():
            with torch.no_grad():
                if self.is_peft:
                    self._log_lora_statistics(model)
                else:
                    self._log_full_weight_statistics(model)
        
        self.gradient_accumulation_counter += 1
        
        return loss.detach() / self.args.gradient_accumulation_steps
    
    def _initialize_lora_tracking(self, model):
        """Initialize tracking for LoRA A and B matrices."""
        self.orig_A = {}
        self.orig_B = {}
        
        # Store original LoRA parameters
        for name, param in model.named_parameters():
            if param.requires_grad and any(kw in name for kw in INCLUDE_KEYWORDS):
                if "lora_A" in name:
                    self.orig_A[name.split("lora_A.")[0]] = param.detach().clone()
                elif "lora_B" in name:
                    self.orig_B[name.split("lora_B.")[0]] = param.detach().clone()
        
        # Register forward hooks for monitoring
        for name, module in model.named_modules():
            if any(kw in name for kw in INCLUDE_KEYWORDS) and isinstance(
                module, LoraLinear
            ):
                hook = get_forward_hook(name)
                module.register_forward_hook(hook)
    
    def _initialize_full_weight_tracking(self, model):
        """Initialize tracking for full weight matrices."""
        self.orig_W = {}
        for name, param in model.named_parameters():
            if param.requires_grad and any(kw in name for kw in INCLUDE_KEYWORDS):
                self.orig_W[name] = param.detach().clone()
    
    def _should_log_statistics(self):
        """Check if we should log statistics at this step."""
        return (
            self.gradient_accumulation_counter % self.args.gradient_accumulation_steps
            == self.args.gradient_accumulation_steps - 1
        )
    
    def _log_lora_statistics(self, model):
        """Log statistics for LoRA parameters (A and B matrices)."""
        A_dict = {}
        B_dict = {}
        
        # Collect current LoRA parameters
        for name, param in model.named_parameters():
            if param.requires_grad and any(kw in name for kw in INCLUDE_KEYWORDS):
                if "lora_A" in name:
                    A_dict[name.split("lora_A.")[0]] = param
                elif "lora_B" in name:
                    B_dict[name.split("lora_B.")[0]] = param
        
        assert len(A_dict) == len(self.orig_A) == len(B_dict) == len(self.orig_B)
        
        # Log statistics for each LoRA module
        for key in A_dict.keys():
            A = A_dict[key]
            B = B_dict[key]
            lora_r = A.shape[0]
            
            # Get gradients and original values
            A_grad = A.grad
            B_grad = B.grad
            A_0 = self.orig_A[key]
            B_0 = self.orig_B[key]
            
            # Compute differences
            A_diff = A - A_0
            B_diff = B - B_0
            BA = torch.matmul(B, A)
            BA_0 = torch.matmul(B_0, A_0)
            BA_diff = BA - BA_0
            
            # Compute norms
            BA_diff_norm = torch.norm(BA_diff).item()
            A_diff_norm = torch.norm(A_diff).item()
            B_diff_norm = torch.norm(B_diff).item()
            A_norm = torch.norm(A).item()
            B_norm = torch.norm(B).item()
            A_grad_norm = torch.norm(A_grad).item()
            B_grad_norm = torch.norm(B_grad).item()
            
            # Singular value analysis
            BA_singular_values = torch.svd_lowrank(BA_diff.float(), q=2 * lora_r)[1][:lora_r]
            top_1_ratio = (BA_singular_values[0] / BA_singular_values.sum()).item()
            top_4_ratio = (BA_singular_values[:4].sum() / BA_singular_values.sum()).item()
            
            # Log to wandb
            wandb.log(
                {
                    f"A_norm/{key}": A_norm,
                    f"B_norm/{key}": B_norm,
                    f"A_grad_norm/{key}": A_grad_norm,
                    f"B_grad_norm/{key}": B_grad_norm,
                    f"A_diff_norm/{key}": A_diff_norm,
                    f"B_diff_norm/{key}": B_diff_norm,
                    f"BA_diff_norm/{key}": BA_diff_norm,
                    f"scaled_BA_diff_norm/{key}": self.scaling * BA_diff_norm,
                    f"BA_top_1_ratio/{key}": top_1_ratio,
                    f"BA_top_4_ratio/{key}": top_4_ratio,
                    "train/global_step": self.state.global_step,
                }
            )
    
    def _log_full_weight_statistics(self, model):
        """Log statistics for full weight matrices (non-LoRA)."""
        W_dict = {}
        
        # Collect weight parameters
        for name, param in model.named_parameters():
            if (
                param.requires_grad
                and any(kw in name for kw in INCLUDE_KEYWORDS)
                and len(param.shape) == 2
            ):
                W_dict[name] = param
        
        # Log statistics for each weight matrix
        for key in W_dict.keys():
            W = W_dict[key]
            W_grad = W.grad
            W_0 = self.orig_W[key]
            W_diff = W - W_0
            
            # Compute norms
            W_diff_norm = torch.norm(W_diff).item()
            W_norm = torch.norm(W).item()
            W_grad_norm = torch.norm(W_grad).item()
            
            # Singular value analysis
            U, S, V = torch.svd(W_diff.float())
            top_1_ratio = (S[0] / S.sum()).item()
            top_4_ratio = (S[:4].sum() / S.sum()).item()
            
            # Log to wandb
            wandb.log(
                {
                    f"W_norm/{key}": W_norm,
                    f"W_grad_norm/{key}": W_grad_norm,
                    f"W_diff_norm/{key}": W_diff_norm,
                    f"W_top_1_ratio/{key}": top_1_ratio,
                    f"W_top_4_ratio/{key}": top_4_ratio,
                    "train/global_step": self.state.global_step,
                }
            )