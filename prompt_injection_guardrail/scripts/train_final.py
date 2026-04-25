from src.models.bert_lora_model import BertLoraClassifier
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

def run_final_training():
    print("Initializing final SOTA training session...")
    clf = BertLoraClassifier()
    clf.train(data_path="data/processed/train_clean.csv")
    print("Training complete.")

if __name__ == "__main__":
    run_final_training()
