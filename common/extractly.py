import re, json, ast

def extract_sentiment(text):
    if not isinstance(text, str):
        return None

    text = text.strip()
    if not text:
        return None

    if 'sentiment' not in text.lower():
        return None

    pattern = r'python\s*(\{.*?\})\s*|(\{.*?\})'
    matches = re.findall(pattern, text, flags=re.DOTALL)

    extracted = []
    for py_block, json_block in matches:
        candidate = py_block or json_block 
        candidate = candidate.strip()

        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                continue

        if isinstance(data, dict) and 'sentiment' in data:
            extracted.append(data['sentiment'])

    if len(extracted) >= 2: 
        return extracted[len(extracted) - 1]
    
    if not extracted:
        return None
    
    return extracted[0] if type(extracted) is list else extracted