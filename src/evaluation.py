import torch
from fire import Fire
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import time

from peft import PeftModel
from accelerate import Accelerator
from utils import initialize_text_to_text_model
from data import DATASET_MAP


def evaluate_model(model, tokenizer, test_dataset, model_name, max_samples=None, device="cuda"):
    """
    Evaluate a single model's performance.
    
    Args:
        model: The model to evaluate
        tokenizer: Associated tokenizer
        test_dataset: Dataset for evaluation
        model_name: Name of the model for logging
        max_samples: Maximum number of samples to evaluate (None for all)
        device: Device to run evaluation on
    
    Returns:
        dict: Evaluation metrics including accuracy and inference time
    """
    model.eval()
    
    # Convert dataset to list to prevent len() errors
    if hasattr(test_dataset, '__iter__') and not isinstance(test_dataset, list):
        test_data = list(test_dataset)
    else:
        test_data = test_dataset
    
    # Limit samples if specified
    if max_samples and len(test_data) > max_samples:
        test_data = test_data[:max_samples]
    
    correct = 0
    total = 0
    inference_times = []
    
    print(f"\nEvaluating {model_name} on {len(test_data)} samples...")
    
    with torch.no_grad():
        for i, sample in enumerate(test_data):
            if i % 20 == 0:
                print(f"  Progress: {i}/{len(test_data)}")
            
            # Prepare input
            input_text = sample['x']
            true_label = sample['y'].lower().strip()  # "different" or "equivalent"
            
            # Tokenization
            inputs = tokenizer(
                input_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(device)
            
            # Measure inference time
            start_time = time.time()
            
            # Generate prediction
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                min_new_tokens=1,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            inference_times.append(time.time() - start_time)
            
            # Decode prediction
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract prediction text
            if "result:" in full_output:
                pred_text = full_output.split("result:")[-1].strip()
            else:
                pred_text = full_output.strip()
            
            pred_label = pred_text.lower().strip()
            
            # Map T5's True/False output to equivalent/different
            if pred_label == "true":
                final_pred = "equivalent"
            elif pred_label == "false":
                final_pred = "different"
            elif pred_label in ["equivalent", "different"]:
                final_pred = pred_label
            else:
                # Partial matching for edge cases
                if any(word in pred_label for word in ["true", "equivalent", "same", "similar", "match"]):
                    final_pred = "equivalent"
                elif any(word in pred_label for word in ["false", "different", "dissimilar", "unrelated"]):
                    final_pred = "different"
                else:
                    final_pred = "different"  # Default for MRPC
            
            # Check correctness
            if final_pred == true_label:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    avg_time = np.mean(inference_times) if inference_times else 0
    
    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_inference_time": avg_time
    }


def load_model(base_model_path, adapter_path, model_name, device="cuda"):
    """
    Load a model with LoRA adapter.
    
    Args:
        base_model_path: Path to base model
        adapter_path: Path to LoRA adapter
        model_name: Name for logging
        device: Device to load model on
    
    Returns:
        tuple: (model, tokenizer) or (None, None) if failed
    """
    print(f"\nLoading {model_name}...")
    
    # Check adapter path exists
    if not os.path.exists(adapter_path):
        print(f"Error: Adapter path does not exist: {adapter_path}")
        return None, None
    
    try:
        # Load base model
        model, tokenizer = initialize_text_to_text_model(
            base_model_path, "ConditionalGeneration", "fp32", flash_attention=False
        )
        print(f"Base model ({base_model_path}) loaded")
        
        # Load adapter
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"Adapter loaded from: {adapter_path}")
        
        model.to(device)
        model.eval()
        print(f"{model_name} ready on {device}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main(
    normal_lora_path: str,
    max_samples: int = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_test_set: bool = False
):
    """
    Evaluate T5 LoRA methods on MRPC dataset.
    
    Args:
        normal_lora_path: Path to LoRA adapter checkpoint
        max_samples: Maximum samples to evaluate (None for all)
        device: Device for evaluation
        use_test_set: Use test set instead of validation set
    """
    
    print("T5 LoRA Evaluation on MRPC")
    print("="*60)
    
    # Load MRPC dataset
    print("Loading MRPC dataset...")
    train_set, val_set, test_set = DATASET_MAP["mrpc"]()
    
    # Select evaluation set
    eval_set = test_set if use_test_set else val_set
    set_name = "test" if use_test_set else "validation"
    
    print(f"Dataset loaded: {len(eval_set)} samples from {set_name} set")
    print("\nNote: T5 generates 'True'/'False' for MRPC")
    print("  'True' -> 'equivalent' (similar sentences)")
    print("  'False' -> 'different' (different sentences)\n")
    
    # Define models to evaluate
    models = [
        {"name": "HiLoRA-v2", "path": normal_lora_path, "color": "blue"},
    ]
    
    results = []
    
    # Evaluate each model
    for model_info in models:
        print(f"\n{'='*40}")
        print(f"Evaluating: {model_info['name']}")
        
        model, tokenizer = load_model("t5-base", model_info["path"], model_info["name"], device)
        
        if model is None:
            continue
        
        result = evaluate_model(model, tokenizer, eval_set, model_info["name"], max_samples, device)
        result["color"] = model_info["color"]
        results.append(result)
        
        print(f"Accuracy: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")
        print(f"Avg Time: {result['avg_inference_time']:.4f}s")
        
        # Clean up memory
        del model, tokenizer
        torch.cuda.empty_cache()
    
    # Output results
    if results:
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS ({set_name.upper()} SET)")
        print(f"{'='*60}")
        print(f"{'Method':<12} {'Accuracy':<10} {'Time(s)':<8}")
        print("-" * 30)
        
        best_acc = 0
        best_model = ""
        
        for r in results:
            print(f"{r['model_name']:<12} {r['accuracy']:<10.4f} {r['avg_inference_time']:<8.4f}")
            if r['accuracy'] > best_acc:
                best_acc = r['accuracy']
                best_model = r['model_name']
        
        print(f"\nBest: {best_model} (Accuracy: {best_acc:.4f})")
        
        # Save visualization
        save_dir = f"./evaluation_results_mrpc_{set_name}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Create comparison plots
        plt.figure(figsize=(10, 5))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        methods = [r["model_name"] for r in results]
        accuracies = [r["accuracy"] for r in results]
        colors = [r["color"] for r in results]
        
        bars = plt.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black')
        plt.title(f"Accuracy Comparison (MRPC {set_name.title()})", fontweight='bold')
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.0)
        
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Time plot
        plt.subplot(1, 2, 2)
        times = [r["avg_inference_time"] for r in results]
        bars = plt.bar(methods, times, color=colors, alpha=0.8, edgecolor='black')
        plt.title("Inference Time", fontweight='bold')
        plt.ylabel("Time (seconds)")
        
        for bar, t in zip(bars, times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.02, 
                    f'{t:.3f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"comparison_mrpc_{set_name}.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save results as JSON
        with open(os.path.join(save_dir, f"results_mrpc_{set_name}.json"), "w") as f:
            json.dump({
                "results": results, 
                "best_model": best_model, 
                "dataset": "mrpc",
                "eval_set": set_name,
                "note": "T5 generates True/False, mapped to equivalent/different"
            }, f, indent=2)
        
        print(f"\nResults saved to: {save_dir}")
        print(f"  Chart: comparison_mrpc_{set_name}.png")
        print(f"  Data: results_mrpc_{set_name}.json")
    
    else:
        print("\nNo models evaluated successfully!")
    
    print(f"\nMRPC evaluation on {set_name} set completed!")


if __name__ == "__main__":
    Fire(main)