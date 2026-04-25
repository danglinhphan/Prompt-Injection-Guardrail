# Constants for the Explainable Prompt Injection Framework

# OWASP Categories
BENIGN = "benign"
INJECTION_DIRECT = "injection-direct"
INJECTION_INDIRECT = "injection-indirect"
DATA_EXFILTRATION = "data-exfiltration"
TOOL_MISUSE = "tool-misuse"

CATEGORIES = [
    BENIGN,
    INJECTION_DIRECT,
    INJECTION_INDIRECT,
    DATA_EXFILTRATION,
    TOOL_MISUSE
]

# Mitigation Actions
ALLOW = "allow"
WARN = "warn"
BLOCK = "block"
SANITIZE = "sanitize"

ACTIONS = [
    ALLOW,
    WARN,
    BLOCK,
    SANITIZE
]

# Thresholds (Adjusted for current model sensitivity)
CONFIDENCE_THRESHOLD_BLOCK = 0.45
CONFIDENCE_THRESHOLD_WARN = 0.30
