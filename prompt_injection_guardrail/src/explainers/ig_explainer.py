import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from peft import PeftModel, PeftConfig
from captum.attr import LayerIntegratedGradients
import os

# Get repository root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, input_ids):
        outputs = self.model(input_ids=input_ids)
        if hasattr(outputs, 'logits'):
            return outputs.logits
        return outputs

class DeepExplainer:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(REPO_ROOT, "models/bert_lora_final")
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DeepExplainer using device: {self.device}")
        
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path {model_path} does not exist. Please train the model first.")

            # Load base model and LoRA weights
            config = PeftConfig.from_pretrained(model_path)
            base_model = DistilBertForSequenceClassification.from_pretrained(
                config.base_model_name_or_path, 
                num_labels=5
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
            self.model.to(self.device)
            self.model.eval()
            
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            
            # Create wrapper for Captum
            self.wrapper = ModelWrapper(self.model)
            
            # Use get_input_embeddings() for robustness
            embedding_layer = self.model.get_input_embeddings()
            self.lig = LayerIntegratedGradients(self.wrapper, embedding_layer)
            print("DeepExplainer initialized successfully.")
        except Exception as e:
            print(f"Error initializing DeepExplainer: {e}")
            self.lig = None # Fallback indicator

    def explain(self, text, target_class_idx):
        if self.lig is None:
            return [{"token": "Explainer not initialized", "score": 0.0}]
            
        try:
            inputs = self.tokenizer.encode_plus(
                text, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            
            input_ids = inputs["input_ids"]
            
            # Calculate attributions
            attributions = self.lig.attribute(
                inputs=input_ids,
                target=int(target_class_idx),
                n_steps=20 # Increased for better quality
            )
            
            # Process attributions
            attributions = attributions.sum(dim=-1).squeeze(0)
            if torch.norm(attributions) > 0:
                attributions = attributions / torch.norm(attributions)
            
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            
            explanation = []
            for token, score in zip(tokens, attributions.tolist()):
                if token not in ['[CLS]', '[SEP]', '[PAD]']:
                    explanation.append({"token": token, "score": float(score)})
                    
            explanation = sorted(explanation, key=lambda x: abs(x['score']), reverse=True)
            return explanation
        except Exception as e:
            print(f"Explanation Error: {e}")
            return [{"token": "Error", "score": 0.0}]

def get_deep_explanation(text, label_idx):
    explainer = DeepExplainer()
    return explainer.explain(text, label_idx)
