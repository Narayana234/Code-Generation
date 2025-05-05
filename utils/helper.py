import re

def clean_code(code):
    """
    Removes unnecessary indentation and special characters.
    """
    return re.sub(r'\s+', ' ', code).strip()
