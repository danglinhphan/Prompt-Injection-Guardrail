from src.core.constants import *
from src.mitigation.sanitizers import sanitize_prompt

class PolicyEngine:
    def __init__(self, block_threshold=CONFIDENCE_THRESHOLD_BLOCK, warn_threshold=CONFIDENCE_THRESHOLD_WARN):
        self.block_threshold = block_threshold
        self.warn_threshold = warn_threshold

    def recommend_action(self, label, confidence):
        """
        Maps a prediction to a mitigation action.
        """
        if label == BENIGN:
            return ALLOW
        
        if confidence >= self.block_threshold:
            return BLOCK
        
        if confidence >= self.warn_threshold:
            if label in [INJECTION_DIRECT, INJECTION_INDIRECT]:
                return SANITIZE
            return WARN
            
        return ALLOW

    def apply_mitigation(self, text, action, explainer=None):
        if action == BLOCK:
            return "[BLOCKED] Potentially malicious prompt detected."
        
        if action == WARN:
            return f"[WARNING] This prompt was flagged as suspicious. Proceed with caution.\n\n{text}"
            
        if action == SANITIZE:
            if explainer and hasattr(explainer, 'explain'):
                try:
                    explanation = explainer.explain(text)
                    # Mask tokens with attribution > 0.1 (risk threshold)
                    words = text.split()
                    sanitized_words = []
                    # Simple mapping of attribution to words (approximated)
                    # In a real system, we would map sub-tokens back to words
                    for i, word in enumerate(words):
                        # This is a simplified demo of XAI-based sanitization
                        # In production, we'd use the exact token offsets
                        if any(attr > 0.2 for token, attr in explanation if token in word):
                            sanitized_words.append("[MASKED]")
                        else:
                            sanitized_words.append(word)
                    return " ".join(sanitized_words)
                except:
                    return sanitize_prompt(text)
            return sanitize_prompt(text)
            
        return text
