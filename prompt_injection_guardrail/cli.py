import argparse
import sys
import os
import torch

# Try to import rich, fallback to standard print if not available
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    USE_RICH = True
except ImportError:
    USE_RICH = False

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.tfidf_model import TfidfClassifier
from src.mitigation.engine import PolicyEngine
from src.core.constants import CATEGORIES
from src.explainers.ig_explainer import DeepExplainer
from src.core.preprocessing import normalize_text

if USE_RICH:
    console = Console()
else:
    class FakeConsole:
        def print(self, msg): print(msg)
        def input(self, msg): return input(msg)
        def status(self, msg):
            class FakeStatus:
                def __enter__(self): print(msg); return self
                def __exit__(self, *args): pass
            return FakeStatus()
    console = FakeConsole()

def display_results(text, label, confidence, action, sanitized_text, top_tokens=None):
    if USE_RICH:
        table = Table(title="🛡️ Guardrail Analysis Results")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Detected Label", f"[bold]{label}[/]")
        table.add_row("Confidence", f"{confidence:.4f}")
        
        action_style = "green" if action == "allow" else "red" if action == "block" else "yellow"
        table.add_row("Action", f"[{action_style}]{action.upper()}[/]")
        
        if sanitized_text != text:
            table.add_row("Sanitized Text", sanitized_text)
            
        console.print(table)
        
        if top_tokens:
            token_table = Table(title="🔍 Risk Attribution (Top Malicious Tokens)")
            token_table.add_column("Token", style="cyan")
            token_table.add_column("Risk Score", style="red")
            for t in top_tokens:
                token_table.add_row(t['token'], f"{t['score']:.4f}")
            console.print(token_table)
    else:
        print("\n" + "="*40)
        print("🛡️ GUARDRAIL ANALYSIS RESULTS")
        print("="*40)
        print(f"Detected Label: {label}")
        print(f"Confidence:     {confidence:.4f}")
        print(f"Action:         {action.upper()}")
        if sanitized_text != text:
            print(f"Sanitized:      {sanitized_text}")
        if top_tokens:
            print("\nRISK ATTRIBUTION (XAI):")
            for t in top_tokens:
                print(f"- {t['token']}: {t['score']:.4f}")
        print("="*40 + "\n")

def run_cli():
    parser = argparse.ArgumentParser(description="🛡️ Prompt-Injection-Guardrail CLI")
    parser.add_argument("text", nargs="?", help="The prompt text to analyze")
    parser.add_argument("--tier", choices=["tier1", "tier2"], default="tier1", help="Detection tier (default: tier1)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Initialize components
    tfidf_classifier = TfidfClassifier()
    policy_engine = PolicyEngine()
    bert_explainer = None

    def analyze(text, tier):
        nonlocal bert_explainer
        
        normalized_text = normalize_text(text)
        
        try:
            if tier == "tier2":
                if bert_explainer is None:
                    with console.status("[bold green]Loading Tier 2 (BERT+LoRA) model...[/]"):
                        bert_explainer = DeepExplainer()
                
                inputs = {k: v.to(bert_explainer.device) for k, v in bert_explainer.tokenizer(normalized_text, return_tensors="pt", padding=True, truncation=True).items()}
                
                with torch.no_grad():
                    outputs = bert_explainer.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    confidence, label_idx = torch.max(probs, dim=-1)
                    label = CATEGORIES[label_idx.item()]
                    confidence = confidence.item()
                
                explanation_data = bert_explainer.explain(normalized_text, label_idx.item())
                top_tokens = explanation_data[:5]
                
                action = policy_engine.recommend_action(label, confidence)
                sanitized_text = policy_engine.apply_mitigation(text, action, explainer=bert_explainer)
                
                display_results(text, label, confidence, action, sanitized_text, top_tokens)
                
            else:
                label, confidence = tfidf_classifier.predict(normalized_text)
                action = policy_engine.recommend_action(label, confidence)
                sanitized_text = policy_engine.apply_mitigation(text, action)
                
                display_results(text, label, confidence, action, sanitized_text)
                
        except Exception as e:
            if USE_RICH:
                console.print(f"[bold red]Error:[/] {str(e)}")
            else:
                print(f"Error: {str(e)}")

    if args.interactive:
        if USE_RICH:
            console.print(Panel("[bold green]🛡️ Prompt-Injection-Guardrail Interactive Mode[/]\nType 'exit' or 'quit' to stop.", subtitle=f"Default Tier: {args.tier}"))
        else:
            print("🛡️ Prompt-Injection-Guardrail Interactive Mode")
            print("Type 'exit' or 'quit' to stop.\n")
            
        while True:
            try:
                user_input = console.input("[bold blue]>>> [/]" if USE_RICH else ">>> ")
                if user_input.lower() in ['exit', 'quit']:
                    break
                if not user_input.strip():
                    continue
                analyze(user_input, args.tier)
            except KeyboardInterrupt:
                break
    elif args.text:
        analyze(args.text, args.tier)
    else:
        parser.print_help()

if __name__ == "__main__":
    run_cli()
