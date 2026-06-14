import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import platform
import torch
import torch.nn as nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

"""
conda  activate termalign
cd ~/CLllm
python pre_experiment/run_baseline.py
"""

# 从其他文件导入模块
from mini_data import get_dataloader, LABELS, ID2LABEL
from token_cl_model import ContrastiveNERModel, TokenSupConLoss
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from config import NERConfig

# 获取配置
config = NERConfig()

def train_epoch(model, dataloader, optimizer, cl_loss_fn, ce_loss_fn, device, alpha=0.5, scheduler=None):
    model.train()
    total_loss = 0
    tqdm_loader = tqdm(dataloader, desc="Training Epoch")
    
    for batch in tqdm_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        logits, proj_features = model(input_ids, attention_mask)
        
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, logits.shape[-1])[active_loss]
        active_labels = labels.view(-1)[active_loss]
        
        valid_mask = active_labels != -100
        active_logits_valid = active_logits[valid_mask]
        active_labels_valid = active_labels[valid_mask]
        
        loss_ce = ce_loss_fn(active_logits_valid, active_labels_valid) if len(active_labels_valid) > 0 else torch.tensor(0.0).to(device)
        
        active_proj = proj_features.view(-1, proj_features.shape[-1])[active_loss]
        cl_features_valid = active_proj[valid_mask]
        
        loss_cl = cl_loss_fn(cl_features_valid, active_labels_valid)
        
        loss = (1 - alpha) * loss_ce + alpha * loss_cl
        
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        tqdm_loader.set_postfix({
            "TotalLoss": f"{loss.item():.4f}", 
            "CE": f"{loss_ce.item() if isinstance(loss_ce, torch.Tensor) else loss_ce:.4f}",
            "CL": f"{loss_cl.item() if isinstance(loss_cl, torch.Tensor) else loss_cl:.4f}"
        })
        
    return total_loss / len(dataloader)


def evaluate(model, dataloader, ce_loss_fn, device, desc="Evaluating"):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits, _ = model(input_ids, attention_mask)
            
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, logits.shape[-1])[active_loss]
            active_labels = labels.view(-1)[active_loss]
            
            valid_mask = active_labels != -100
            active_logits_valid = active_logits[valid_mask]
            active_labels_valid = active_labels[valid_mask]
            
            if len(active_labels_valid) > 0:
                loss_ce = ce_loss_fn(active_logits_valid, active_labels_valid)
                total_loss += loss_ce.item()
            
            preds = torch.argmax(logits, dim=2)
            for i in range(labels.shape[0]):
                p = preds[i]
                l = labels[i]
                mask = l != -100
                
                pred_ids = p[mask].cpu().numpy().tolist()
                label_ids = l[mask].cpu().numpy().tolist()
                
                # 转换为真实的标签进行 span 级别的评估
                pred_labels = [ID2LABEL[idx] for idx in pred_ids]
                true_labels = [ID2LABEL[idx] for idx in label_ids]
                
                all_preds.append(pred_labels)
                all_labels.append(true_labels)
                
    try:
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds)
    except Exception as e:
        precision, recall, f1, report = 0.0, 0.0, 0.0, str(e)
    
    return total_loss / len(dataloader), f1, recall, precision, report


def main():
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    is_macos = platform.system() == "Darwin"
    data_limits = {"train": None, "val": None, "test": None}
    if is_macos:
        data_limits = {"train": 10, "val": 5, "test": 1}
        print("macOS detected: using tiny datasets (train=10, val=5, test=1).")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    
    print("Loading datasets (Train, Val, Test)...")
    train_loader = get_dataloader(
        config.train_data_path,
        tokenizer,
        batch_size=config.batch_size,
        shuffle=True,
        max_len=config.max_len,
        limit=data_limits["train"],
    )
    val_loader = get_dataloader(
        config.val_data_path,
        tokenizer,
        batch_size=config.batch_size,
        shuffle=False,
        max_len=config.max_len,
        limit=data_limits["val"],
    )
    test_loader = get_dataloader(
        config.test_data_path,
        tokenizer,
        batch_size=config.batch_size,
        shuffle=False,
        max_len=config.max_len,
        limit=data_limits["test"],
    )
    
    print("Initializing Model...")
    model = ContrastiveNERModel(config.model_name_or_path, num_labels=config.num_labels, proj_dim=config.proj_dim).to(device)
    
    optimizer = AdamW(model.parameters(), lr=config.lr)
    
    epochs = config.epochs
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    
    cl_loss_fn = TokenSupConLoss(temperature=config.temperature)
    ce_loss_fn = nn.CrossEntropyLoss()
    
    alpha = config.alpha  # CE loss vs Contrastive loss
    
    print("Starting training loop...")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, cl_loss_fn, ce_loss_fn, device, alpha=alpha, scheduler=scheduler)
        val_loss, val_f1, val_recall, val_precision, val_report = evaluate(model, val_loader, ce_loss_fn, device, desc="Validation")
        
        print(f"==> Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"    Val Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}\n")
        
    print("Training finished! Testing model...")
    test_loss, test_f1, test_recall, test_precision, test_report = evaluate(model, test_loader, ce_loss_fn, device, desc="Testing")
    print(f"==> Test Loss: {test_loss:.4f}")
    print(f"    Test Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}")
    print("\nTest Classification Report:")
    print(test_report)

if __name__ == "__main__":
    main()
