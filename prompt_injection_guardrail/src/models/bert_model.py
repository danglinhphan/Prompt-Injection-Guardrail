import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
import os

# Get repository root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

class PromptDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = {
            "benign": 0,
            "injection-direct": 1,
            "injection-indirect": 2,
            "data-exfiltration": 3,
            "tool-misuse": 4
        }

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.label_map[self.labels[item]]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_bert_lora():
    print("Initializing DistilBERT Multilingual + LoRA Training...")
    
    # Load dataset
    data_path = os.path.join(REPO_ROOT, "data/processed/dataset.csv")
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    
    # Initialize tokenizer and model (Multilingual for SOTA support)
    model_name = "distilbert-base-multilingual-cased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=5)
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16, # Increased rank for multilingual complexity
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["attention.q_lin", "attention.k_lin", "attention.v_lin", "attention.out_lin"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Prepare data (Simple split for demo)
    train_size = int(0.9 * len(df))
    df_train = df.iloc[:train_size].reset_index(drop=True)
    df_val = df.iloc[train_size:].reset_index(drop=True)
    
    train_dataset = PromptDataset(df_train['text'], df_train['label'], tokenizer)
    val_dataset = PromptDataset(df_val['text'], df_val['label'], tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(REPO_ROOT, "models/bert_lora_temp"),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=200,
        weight_decay=0.01,
        logging_dir=os.path.join(REPO_ROOT, "logs"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    print("Training started...")
    trainer.train()
    
    # Save model and tokenizer
    save_path = os.path.join(REPO_ROOT, "models/bert_lora_final")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_bert_lora()
