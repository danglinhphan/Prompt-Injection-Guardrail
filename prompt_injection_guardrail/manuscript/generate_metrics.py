import pandas as pd
import torch
import os
import sys
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from tqdm import tqdm

# Get repository root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from models.tfidf_model import TfidfClassifier
from explainers.ig_explainer import DeepExplainer
from core.preprocessing import normalize_text

def generate_metrics():
    data_path = "data/processed/dataset.csv"
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        return

    print("Loading data...")
    # Load clean holdout test set (no overlap with training)
    test_df_path = os.path.join(REPO_ROOT, "data/processed/test_holdout.csv")
    test_df = pd.read_csv(test_df_path)
    # Take a representative sample for the table if it's too large, or use full
    # Use full test set
    
    # Tier 1 Evaluation
    print("Evaluating Tier 1 (TF-IDF)...")
    t1 = TfidfClassifier()
    y_true = test_df['label']
    y_pred_t1 = []
    for text in tqdm(test_df['text']):
        label, _ = t1.predict(normalize_text(text))
        y_pred_t1.append(label)
    
    t1_metrics = precision_recall_fscore_support(y_true, y_pred_t1, average='weighted')
    
    # Tier 2 Evaluation
    print("Evaluating Tier 2 (DistilBERT+LoRA)...")
    try:
        t2 = DeepExplainer()
        y_pred_t2 = []
        CATEGORIES = ["benign", "injection-direct", "injection-indirect", "data-exfiltration", "tool-misuse"]
        
        # Ensure we are actually using the LoRA weights
        print(f"Model type: {type(t2.model)}")
        
        for text in tqdm(test_df['text']):
            norm_text = normalize_text(text)
            inputs = {k: v.to(t2.device) for k, v in t2.tokenizer(norm_text, return_tensors="pt", padding=True, truncation=True).items()}
            with torch.no_grad():
                outputs = t2.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                label_idx = torch.argmax(probs, dim=-1).item()
                y_pred_t2.append(CATEGORIES[label_idx])
        
        t2_metrics = precision_recall_fscore_support(y_true, y_pred_t2, average='weighted')
    except Exception as e:
        print(f"Tier 2 evaluation failed: {e}")
        t2_metrics = None

    # Generate LaTeX table
    table_content = """
\\begin{table}[H]
\\caption{Comparative Performance Metrics of Tier 1 and Tier 2 Guardrails}
\\begin{center}
\\resizebox{\\columnwidth}{!}{
\\begin{tabular}{lcccccc}
\\toprule
\\textbf{Model Tier} & \\textbf{Acc.} & \\textbf{Prec.} & \\textbf{Rec.} & \\textbf{F1} & \\textbf{FPR} & \\textbf{FNR} \\\\
\\midrule
"""
    # Helper to calculate FPR/FNR from confusion matrix
    def get_rates(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=["benign", "injection-direct", "injection-indirect", "data-exfiltration", "tool-misuse"])
        # Benign is index 0
        fp = cm[0, 1:].sum()
        tn = cm[0, 0]
        fn = cm[1:, 0].sum()
        tp = cm[1:, 1:].sum()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        return fpr, fnr

    fpr1, fnr1 = get_rates(y_true, y_pred_t1)
    acc1 = accuracy_score(y_true, y_pred_t1)
    table_content += f"Tier 1 (TF-IDF+LR) & {acc1:.3f} & {t1_metrics[0]:.3f} & {t1_metrics[1]:.3f} & {t1_metrics[2]:.3f} & {fpr1:.3f} & {fnr1:.3f} \\\\\n"
    
    if t2_metrics:
        fpr2, fnr2 = get_rates(y_true, y_pred_t2)
        acc2 = accuracy_score(y_true, y_pred_t2)
        prec2, rec2, f1_2 = t2_metrics[0], t2_metrics[1], t2_metrics[2]
    else:
        acc2, prec2, rec2, f1_2, fpr2, fnr2 = 0, 0, 0, 0, 0, 0
            
    table_content += f"Tier 2 (BERT+LoRA) & {acc2:.3f} & {prec2:.3f} & {rec2:.3f} & {f1_2:.3f} & {fpr2:.3f} & {fnr2:.3f} \\\\\n"
    
    table_content += """
\\bottomrule
\\end{tabular}
}
\\label{tab:comparative_metrics}
\\\\ \\scriptsize{Note: Acc=Accuracy, Prec=Precision, Rec=Recall. FPR/FNR relative to 'benign' class.}
\\end{center}
\\end{table}
"""
    
    output_path = os.path.join(REPO_ROOT, "manuscript/tables/results.tex")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(table_content)
    
    print(f"Metrics generated and saved to {output_path}")

if __name__ == "__main__":
    generate_metrics()
