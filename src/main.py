import torch
from fire import Fire
import matplotlib.pyplot as plt
import csv
import os

from peft import PeftModel, LoraConfig, get_peft_model
from accelerate import Accelerator
from utils import (
    transform_dataset,
    initialize_text_to_text_model,
    find_all_linear_modules,
    train_text_to_text_model,
)
from data import DATASET_MAP
import wandb


def collect_activations(model, dataset, n_samples=64):
    """
    Collect input activations from Linear layers.
    
    Returns:
        dict: {layer_name: tensor[n_tokens, in_features]}
    """
    activations = {}
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(input, tuple):
                h = input[0]
            else:
                h = input
            
            h = h.detach().cpu()
            
            if len(h.shape) == 3:
                h_flat = h.reshape(-1, h.shape[-1])
            elif len(h.shape) == 2:
                h_flat = h
            else:
                return
            
            if h_flat.shape[-1] >= 64:
                if name not in activations:
                    activations[name] = []
                activations[name].append(h_flat)
        
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if 'embed' not in name.lower() and 'lm_head' not in name.lower():
                hooks.append(module.register_forward_hook(make_hook(name)))
    
    model.eval()
    model.cpu()
    
    print(f"Collecting activations from {n_samples} samples...")
    with torch.no_grad():
        for i, example in enumerate(dataset):
            if i >= n_samples:
                break
            
            input_ids = example['input_ids'].unsqueeze(0)
            attention_mask = example['attention_mask'].unsqueeze(0)
            labels = example['labels'].unsqueeze(0)
            
            try:
                model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            except Exception as e:
                print(f"Warning: Forward pass failed at sample {i}: {e}")
                continue
    
    for h in hooks:
        h.remove()
    
    result = {}
    for name, acts in activations.items():
        if len(acts) > 0:
            result[name] = torch.cat(acts, dim=0)
    
    print(f"Collected activations from {len(result)} modules")
    return result


def collect_gradients(model, dataset, n_samples=64):
    """
    Collect output gradients from Linear layers.
    Each sample's gradient is preserved separately (not averaged).
    
    Returns:
        dict: {layer_name: tensor[n_samples, out_features]}
    """
    gradients = {}
    
    def make_hook(name):
        def hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                g = grad_output[0]
            else:
                g = grad_output
            
            if g is None:
                return
            
            g = g.detach().cpu()
            
            # Average over batch and sequence dimensions
            if len(g.shape) == 3:
                g_avg = g.mean(dim=[0, 1])
            elif len(g.shape) == 2:
                g_avg = g.mean(dim=0)
            else:
                return
            
            if name not in gradients:
                gradients[name] = []
            gradients[name].append(g_avg)
        
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if 'embed' not in name.lower() and 'lm_head' not in name.lower():
                hooks.append(module.register_full_backward_hook(make_hook(name)))
    
    model.train()
    model.cpu()
    
    print(f"Collecting gradients from {n_samples} samples...")
    for i, example in enumerate(dataset):
        if i >= n_samples:
            break
        
        input_ids = example['input_ids'].unsqueeze(0)
        attention_mask = example['attention_mask'].unsqueeze(0)
        labels = example['labels'].unsqueeze(0)
        
        try:
            model.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
        except Exception as e:
            print(f"Warning: Backward pass failed at sample {i}: {e}")
            continue
    
    for h in hooks:
        h.remove()
    
    result = {}
    for name, grads in gradients.items():
        if len(grads) > 0:
            result[name] = torch.stack(grads, dim=0)
    
    print(f"Collected gradients from {len(result)} modules")
    return result


def compute_hilora_init_dual_svd(activations, gradients, rank, scale=0.01):
    """
    Dual SVD initialization for HiLoRA.
    
    Key idea:
    - A: Activation SVD captures input space principal directions
    - B: Gradient SVD captures output space principal directions
    
    Args:
        activations: {name: tensor[n_tokens, in_features]}
        gradients: {name: tensor[n_samples, out_features]}
        rank: LoRA rank
        scale: Scaling factor for B matrix
    
    Returns:
        dict: {name: {'A': tensor[rank, in], 'B': tensor[out, rank]}}
    """
    hilora_init = {}
    
    for name, H in activations.items():
        if name not in gradients:
            continue
        
        G = gradients[name]
        
        try:
            in_features = H.shape[1]
            out_features = G.shape[1]
            n_samples = G.shape[0]
            
            # A: Activation SVD
            # Top-r right singular vectors of H capture input space principal directions
            q_h = min(2 * rank, H.shape[0], H.shape[1])
            U_h, S_h, V_h = torch.svd_lowrank(H.float(), q=q_h)
            A = V_h[:, :rank].T.contiguous()
            
            # B: Gradient SVD
            # Top-r right singular vectors of G capture output space gradient directions
            q_g = min(2 * rank, n_samples, out_features)
            U_g, S_g, V_g = torch.svd_lowrank(G.float(), q=q_g)
            
            # Weight by singular values (importance of each direction)
            V_g_topk = V_g[:, :rank]
            S_g_topk = S_g[:rank]
            S_g_normalized = S_g_topk / (S_g_topk.sum() + 1e-8) * rank
            
            B = V_g_topk * S_g_normalized.unsqueeze(0) * scale
            B = B.contiguous()
            
            hilora_init[name] = {'A': A, 'B': B}
            
        except Exception as e:
            print(f"Warning: Dual SVD failed for {name}: {e}")
            continue
    
    print(f"Computed HiLoRA initialization for {len(hilora_init)} modules")
    return hilora_init


def apply_hilora_init(model, hilora_init):
    """
    Apply computed initialization to LoRA parameters.
    
    Returns:
        int: Number of successfully initialized modules
    """
    count = 0
    total = 0
    
    for name, module in model.named_modules():
        if not hasattr(module, 'lora_A'):
            continue
        
        total += 1
        
        expected_A = module.lora_A['default'].weight.shape
        expected_B = module.lora_B['default'].weight.shape
        
        clean_name = name.replace('base_model.model.', '')
        
        matched = None
        for init_name, init_dict in hilora_init.items():
            if clean_name in init_name or init_name in clean_name:
                if (init_dict['A'].shape == expected_A and 
                    init_dict['B'].shape == expected_B):
                    matched = init_dict
                    break
        
        if matched is None:
            continue
        
        device = module.lora_A['default'].weight.device
        module.lora_A['default'].weight.data = matched['A'].to(device)
        module.lora_B['default'].weight.data = matched['B'].to(device)
        count += 1
    
    print(f"Applied HiLoRA initialization to {count}/{total} LoRA modules")
    return count


def analyze_gradient_svd(gradients, rank, top_k=3):
    """
    Analyze gradient SVD statistics for visualization/debugging.
    """
    print("\n" + "="*60)
    print("Gradient SVD Analysis")
    print("="*60)
    
    for i, (name, G) in enumerate(gradients.items()):
        if i >= top_k:
            break
        
        n_samples, out_features = G.shape
        q = min(2 * rank, n_samples, out_features)
        
        U, S, V = torch.svd_lowrank(G.float(), q=q)
        
        total_var = (S ** 2).sum()
        explained_var = (S[:rank] ** 2).sum() / total_var * 100
        
        print(f"\n[{name}]")
        print(f"  Shape: {G.shape}")
        print(f"  Top-{rank} singular values: {S[:rank].tolist()[:5]}...")
        print(f"  Explained variance (top-{rank}): {explained_var:.1f}%")
        print(f"  S[0]/S[1] ratio: {S[0]/S[1]:.2f}x")


def main(
    lora_alpha: int = 16,
    lora_rank: int = 8,
    n_samples: int = 512,
    scale: float = 0.03,
    seed: int = 7,
    analyze: bool = False,
):
    """
    HiLoRA: Dual SVD Initialization
    
    Args:
        lora_alpha: LoRA alpha parameter
        lora_rank: LoRA rank
        n_samples: Number of samples for activation/gradient collection
        scale: B matrix initialization scale
        seed: Random seed
        analyze: Whether to print gradient SVD analysis
    """
    accelerator = Accelerator()
    
    config = dict(
        model="t5-base",
        dataset="mrpc",
        alpha=lora_alpha,
        rank=lora_rank,
        n_samples=n_samples,
        scale=scale,
        seed=seed,
        version="dual_svd",
    )
    
    wandb_name = "_".join([f"{k}={v}" for k, v in config.items()])
    
    if accelerator.is_local_main_process:
        wandb.init(name=wandb_name, mode="offline", group="hilora", project="HiLoRA")
        print("="*70)
        print("HiLoRA: Dual SVD Initialization")
        print("  A = Activation SVD (input space)")
        print("  B = Gradient SVD (output space)")
        print("="*70)
    
    # Load model and dataset
    model, tokenizer = initialize_text_to_text_model(
        "t5-base", "ConditionalGeneration", "fp32", flash_attention=False
    )
    
    dataset_func = DATASET_MAP["mrpc"]
    train_set, val_set, _ = dataset_func()
    
    transform_dataset("ConditionalGeneration", tokenizer, train_set, 128)
    transform_dataset("ConditionalGeneration", tokenizer, val_set, 128)
    
    # Step 1: Collect activations
    if accelerator.is_local_main_process:
        print("\n[Step 1/4] Collecting Activations")
    
    activations = collect_activations(model, train_set, n_samples)
    
    if len(activations) == 0:
        print("Error: No activations collected!")
        return
    
    # Step 2: Collect gradients
    if accelerator.is_local_main_process:
        print("\n[Step 2/4] Collecting Gradients")
    
    gradients = collect_gradients(model, train_set, n_samples)
    
    if len(gradients) == 0:
        print("Error: No gradients collected!")
        return
    
    # Optional: Analyze gradient SVD
    if analyze and accelerator.is_local_main_process:
        analyze_gradient_svd(gradients, lora_rank)
    
    # Step 3: Compute dual SVD initialization
    if accelerator.is_local_main_process:
        print("\n[Step 3/4] Computing Dual SVD Initialization")
    
    hilora_init = compute_hilora_init_dual_svd(activations, gradients, lora_rank, scale)
    
    if len(hilora_init) == 0:
        print("Error: No HiLoRA initialization computed!")
        return
    
    # Step 4: Create LoRA model and apply initialization
    if accelerator.is_local_main_process:
        print("\n[Step 4/4] Applying HiLoRA Initialization")
    
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=find_all_linear_modules(model),
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    
    model = get_peft_model(model, lora_config)
    n_initialized = apply_hilora_init(model, hilora_init)
    
    if n_initialized == 0:
        print("Error: Failed to initialize any modules!")
        return
    
    # Save initial checkpoint
    save_dir = f"./snapshot_hilora/{wandb_name}"
    if accelerator.is_local_main_process:
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        print(f"\nSaved initial checkpoint to {save_dir}")
    
    # Training
    if accelerator.is_local_main_process:
        print("\nTraining...")
    
    loss_log = []
    
    def log_loss_callback(loss, step):
        loss_log.append((step, loss))
        if accelerator.is_local_main_process:
            wandb.log({"loss": loss, "step": step})
    
    model = train_text_to_text_model(
        run_name=f"peft_test/{wandb_name}",
        train_dataset=train_set,
        valid_dataset=val_set,
        model=model,
        tokenizer=tokenizer,
        model_type="ConditionalGeneration",
        num_train_epochs=8,
        per_device_batch_size=32,
        real_batch_size=32 * accelerator.num_processes,
        bf16=False,
        eval_epochs=1,
        early_stopping_patience=5,
        max_length=128,
        logging_steps=1,
        learning_rate=1e-4,
        num_process=accelerator.num_processes,
        gradient_checkpointing=False,
        seed=seed,
        training_args=dict(
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
            warmup_ratio=0.03,
            weight_decay=0.0,
        ),
        log_loss_callback=log_loss_callback,
    )
    
    # Save results
    if accelerator.is_local_main_process:
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        
        # Save loss history
        with open(f"{save_dir}/loss_history.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss"])
            writer.writerows(loss_log)
        
        # Plot loss curve
        if len(loss_log) > 0:
            steps, losses = zip(*loss_log)
            plt.figure(figsize=(10, 6))
            plt.plot(steps, losses, label="HiLoRA(Dual SVD)", linewidth=2, color='#2E86AB')
            plt.xlabel("Step", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.title("HiLoRA: Dual SVD Initialization", fontsize=14, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/loss_curve.png", dpi=150)
            plt.close()
            print("Loss curve saved")
        
        print("\nTraining complete!")
        print(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    Fire(main)