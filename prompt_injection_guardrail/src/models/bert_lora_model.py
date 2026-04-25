import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd
from datasets import Dataset
import os

class BertLoraClassifier:
    def __init__(self, model_name="distilbert-base-multilingual-cased", model_path="models/bert_lora_final"):
        self.model_name = model_name
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.label2id = {"benign": 0, "injection-direct": 1, "injection-indirect": 2, "data-exfiltration": 3, "tool-misuse": 4}
        self.id2label = {v: k for k, v in self.label2id.items()}

    def train(self, data_path="data/processed/dataset.csv"):
        print(f"Training Tier 2 ({self.model_name} + LoRA)...")
        df = pd.read_csv(data_path)
        df['label'] = df['label'].map(self.label2id)
        
        # Split into train/val
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
        
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        
        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_val = val_dataset.map(tokenize_function, batched=True)
        
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=5, id2label=self.id2label, label2id=self.label2id
        )
        
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, 
            inference_mode=False, 
            r=8, 
            lora_alpha=16, 
            lora_dropout=0.1,
            target_modules=["q_lin", "v_lin"] # Specific for DistilBERT
        )
        
        self.model = get_peft_model(model, peft_config)
        self.model.print_trainable_parameters()
        
        training_args = TrainingArguments(
            output_dir=self.model_path,
            learning_rate=2e-4, 
            per_device_train_batch_size=4, 
            per_device_eval_batch_size=4,
            num_train_epochs=1, 
            weight_decay=0.01,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="no",
            logging_steps=100,
            fp16=True if torch.cuda.is_available() else False, 
            push_to_hub=False,
            report_to="none"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=self.tokenizer,
        )
        
        trainer.train()
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        print(f"Model saved to {self.model_path}")

    def predict(self, text):
        if self.model is None:
            # Load model if exists
            from peft import PeftModel
            base_model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=5)
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
        
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        probs = torch.softmax(logits, dim=1).numpy()[0]
        idx = probs.argmax()
        label = self.id2label[idx]
        confidence = probs[idx]
        return label, confidence

if __name__ == "__main__":
    # Note: Training might be skipped if no GPU or if environment is restricted
    # But we provide the code as requested.
    clf = BertLoraClassifier()
    # clf.train() 
