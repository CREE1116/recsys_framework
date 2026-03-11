import os

directories = ['src', 'scripts', 'configs', 'data', 'aspire_experiments']

def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return False

    lines = content.split('\n')
    changed = False
    
    for i, line in enumerate(lines):
        # Find "open("
        idx = line.find('open(')
        
        # Avoid matching non-function calls like variable names ending in open
        if idx > 0 and line[idx-1].isalnum():
            continue
            
        if idx != -1:
            # If 'encoding=' is already somewhere in the line, skip
            if 'encoding=' in line or 'encoding =' in line:
                continue
            # If binary mode, skip
            if "'rb'" in line or '"rb"' in line or "'wb'" in line or '"wb"' in line or "'ab'" in line or '"ab"' in line:
                continue
                
            # Find the matching closing parenthesis
            p_count = 0
            end_idx = -1
            for j in range(idx + 4, len(line)):
                if line[j] == '(':
                    p_count += 1
                elif line[j] == ')':
                    p_count -= 1
                    if p_count == 0:
                        end_idx = j
                        break
            
            if end_idx != -1:
                inner = line[idx+5:end_idx]
                if 'encoding' not in inner:
                    new_line = line[:end_idx] + ", encoding='utf-8'" + line[end_idx:]
                    lines[i] = new_line
                    changed = True

    if changed:
        new_content = '\n'.join(lines)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    return False

modified = 0
for d in directories:
    if not os.path.exists(d): continue
    for root, _, files in os.walk(d):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                if process_file(path):
                    modified += 1
                    print(f"Updated {path}")

print(f"Total files updated: {modified}")
