"""
Paper 1 - Transformer fine-tuning experiments for log anomaly detection.
Uses synthetic data if real datasets not yet downloaded.
Fine-tunes BERT, RoBERTa, DeBERTa on log text for binary anomaly classification.
"""

import json
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_log_texts(n_samples=2000):
    """Generate synthetic log messages for transformer training."""
    np.random.seed(42)

    normal_templates = [
        "PacketResponder {blk} for block {blk} terminating",
        "Receiving block {blk} src: {ip} dest: {ip}",
        "BLOCK* NameSystem.allocateBlock: {path}. {blk}",
        "Verification succeeded for {blk}",
        "Served block {blk} to {ip}",
        "Received block {blk} of size {size} from {ip}",
        "BLOCK* NameSystem.addStoredBlock: blockMap updated: {ip} is added to {blk}",
        "Deleting block {blk} file /mnt/hadoop/dfs/data/current/{blk}",
    ]

    anomaly_templates = [
        "ERROR: BlockReport failed for block {blk}: connection reset",
        "FATAL: Unable to allocate new block for {path}: disk full",
        "WARN: Block {blk} is CORRUPT on {ip}, marking for replication",
        "ERROR: Pipeline failed for block {blk}: timeout after 60000ms",
        "FATAL: DataNode {ip} is dead, removing from cluster",
        "ERROR: Replication failed for {blk}: no available datanodes",
        "WARN: NameNode safemode: blocks missing, expected {n} received {m}",
        "ERROR: Heartbeat timeout from {ip}, marking node as stale",
    ]

    texts = []
    labels = []

    n_anomaly = int(n_samples * 0.15)
    n_normal = n_samples - n_anomaly

    for _ in range(n_normal):
        # Build a log session (5-15 messages)
        n_msgs = np.random.randint(5, 15)
        session = []
        for _ in range(n_msgs):
            template = np.random.choice(normal_templates)
            msg = template.format(
                blk=f"blk_{np.random.randint(1000000, 9999999)}",
                ip=f"10.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(1,255)}",
                path=f"/user/root/data/part-{np.random.randint(0,999):05d}",
                size=np.random.randint(10000, 99999999),
                n=np.random.randint(100, 999),
                m=np.random.randint(100, 999),
            )
            session.append(msg)
        texts.append(" [SEP] ".join(session))
        labels.append(0)

    for _ in range(n_anomaly):
        n_msgs = np.random.randint(5, 15)
        session = []
        # Mix normal and anomaly messages
        n_anomaly_msgs = np.random.randint(1, min(4, n_msgs))
        for i in range(n_msgs):
            if i < n_anomaly_msgs:
                template = np.random.choice(anomaly_templates)
            else:
                template = np.random.choice(normal_templates)
            msg = template.format(
                blk=f"blk_{np.random.randint(1000000, 9999999)}",
                ip=f"10.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(1,255)}",
                path=f"/user/root/data/part-{np.random.randint(0,999):05d}",
                size=np.random.randint(10000, 99999999),
                n=np.random.randint(100, 999),
                m=np.random.randint(100, 999),
            )
            session.append(msg)
        np.random.shuffle(session)
        texts.append(" [SEP] ".join(session))
        labels.append(1)

    # Shuffle
    idx = np.random.permutation(n_samples)
    texts = [texts[i] for i in idx]
    labels = [labels[i] for i in idx]

    return texts, labels


def fine_tune_and_evaluate(model_name, texts_train, y_train, texts_test, y_test,
                            max_length=256, batch_size=16, epochs=3, lr=2e-5):
    """Fine-tune a transformer and evaluate."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.utils.data import DataLoader, TensorDataset
    from torch.optim import AdamW

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Load model and tokenizer
    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    print(f"  Parameters: {n_params/1e6:.1f}M, Size: {model_size_mb:.1f}MB")

    # Tokenize
    print(f"  Tokenizing {len(texts_train)} train + {len(texts_test)} test samples...")
    train_enc = tokenizer(texts_train, truncation=True, padding=True,
                         max_length=max_length, return_tensors='pt')
    test_enc = tokenizer(texts_test, truncation=True, padding=True,
                        max_length=max_length, return_tensors='pt')

    train_dataset = TensorDataset(
        train_enc['input_ids'], train_enc['attention_mask'],
        torch.tensor(y_train, dtype=torch.long)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(
        test_enc['input_ids'], test_enc['attention_mask'],
        torch.tensor(y_test, dtype=torch.long)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr)

    # Training
    print(f"  Training for {epochs} epochs...")
    train_start = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            ids, mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(train_loader)
        print(f"    Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")
    train_time = time.time() - train_start

    # Evaluation
    model.eval()
    all_preds = []
    all_probs = []

    eval_start = time.time()
    with torch.no_grad():
        for batch in test_loader:
            ids, mask, _ = [b.to(device) for b in batch]
            outputs = model(input_ids=ids, attention_mask=mask)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    eval_time = time.time() - eval_start
    inference_ms = eval_time / len(y_test) * 1000

    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = {
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'auc': float(roc_auc_score(y_test, y_prob)),
        'inference_ms': float(inference_ms),
        'model_size_mb': float(model_size_mb),
        'n_params_millions': float(n_params / 1e6),
        'train_time_s': float(train_time),
    }

    return metrics


def main():
    print("=" * 60)
    print("Paper 1: Transformer Fine-Tuning Experiments")
    print("=" * 60)

    # Generate data
    texts, labels = generate_synthetic_log_texts(n_samples=2000)
    texts_train, texts_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train: {len(texts_train)}, Test: {len(texts_test)}")
    print(f"Anomaly rate: {np.mean(labels)*100:.1f}%")

    models = {
        'BERT-base': 'bert-base-uncased',
        'DistilBERT': 'distilbert-base-uncased',
        # 'RoBERTa-base': 'roberta-base',        # uncomment for full benchmark
        # 'DeBERTa-v3': 'microsoft/deberta-v3-base',  # uncomment for full benchmark
    }

    all_results = {}

    for name, model_id in models.items():
        print(f"\n{'='*50}")
        print(f"Model: {name} ({model_id})")
        print(f"{'='*50}")

        try:
            metrics = fine_tune_and_evaluate(
                model_id, texts_train, y_train, texts_test, y_test,
                max_length=256, batch_size=16, epochs=3
            )
            all_results[name] = metrics
            print(f"\n  RESULTS:")
            print(f"    F1:        {metrics['f1']*100:.1f}%")
            print(f"    Precision: {metrics['precision']*100:.1f}%")
            print(f"    Recall:    {metrics['recall']*100:.1f}%")
            print(f"    AUC:       {metrics['auc']:.4f}")
            print(f"    Latency:   {metrics['inference_ms']:.2f} ms/sample")
            print(f"    Size:      {metrics['model_size_mb']:.1f} MB")
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[name] = {'error': str(e)}

    # Save results
    output_file = RESULTS_DIR / "transformer_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[SAVED] Results -> {output_file}")

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'F1':>8} {'AUC':>8} {'Size(MB)':>10} {'Latency(ms)':>12}")
    print("-" * 58)
    for name, m in all_results.items():
        if 'error' not in m:
            print(f"{name:<20} {m['f1']*100:>7.1f}% {m['auc']:>7.4f} {m['model_size_mb']:>9.1f} {m['inference_ms']:>11.2f}")


if __name__ == "__main__":
    main()
