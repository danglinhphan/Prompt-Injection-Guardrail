import re

def strip_instruction_overrides(text):
    """
    Strips common prompt injection triggers in English and Vietnamese.
    """
    triggers = [
        # English
        r"ignore (all )?previous instructions",
        r"system prompt override",
        r"disregard (all )?safety (filters|settings)",
        r"you are now (an? )?(evil|unfiltered|neutral) AI",
        r"forget everything (I said )?before",
        r"reveal your (system |secret )?prompt",
        r"new rule:",
        # Vietnamese
        r"bỏ qua (tất cả )?chỉ dẫn trước đó",
        r"ghi đè lệnh hệ thống",
        r"quên hết (những gì tôi đã nói )?trước đó",
        r"mày là (một )?AI (xấu xa|trung lập)",
        r"tiết lộ câu lệnh hệ thống"
    ]
    
    sanitized_text = text
    for pattern in triggers:
        sanitized_text = re.sub(pattern, "[STRIPPED_INSTRUCTION]", sanitized_text, flags=re.IGNORECASE)
    
    return sanitized_text

def disable_tool_calls(text):
    """
    Disables patterns that look like tool or command execution.
    """
    tool_patterns = [
        r"run command:",
        r"execute:",
        r"shell:",
        r"rm -rf",
        r"delete user",
        r"chạy lệnh:",
        r"thực thi:"
    ]
    sanitized_text = text
    for pattern in tool_patterns:
        sanitized_text = re.sub(pattern, "[TOOL_CALL_DISABLED]", sanitized_text, flags=re.IGNORECASE)
    return sanitized_text

def sanitize_prompt(text):
    text = strip_instruction_overrides(text)
    text = disable_tool_calls(text)
    return text
