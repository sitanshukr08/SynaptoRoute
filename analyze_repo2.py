import os
import ast
import json

def extract_docstring(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        if filepath.endswith('.py'):
            tree = ast.parse(content)
            return ast.get_docstring(tree) or ""
        elif filepath.endswith('.md'):
            # First few lines
            return "\n".join(content.split('\n')[:5]).strip()
        else:
            return ""
    except:
        return ""

def analyze_file(filepath, base_dir):
    rel_path = os.path.relpath(filepath, base_dir).replace('\\', '/')
    purpose = extract_docstring(filepath)
    imports = []
    exports = []
    
    if filepath.endswith('.py'):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module if node.module else ""
                    imports.append(module)
                    for alias in node.names:
                        exports.append(alias.name)
                elif isinstance(node, ast.ClassDef) or isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    exports.append(node.name)
        except Exception as e:
            purpose = f"Error parsing: {str(e)}"

    return {
        "path": rel_path,
        "purpose": purpose.strip() if purpose else "No explicit purpose documented.",
        "imports": list(set(imports)),
        "exports": list(set(exports)),
    }

def main():
    base_dir = r"c:\Users\sitan\OneDrive\Desktop\synaptoroute"
    
    results = []
    # Only walk src/, tests/, docs/, and root
    dirs_to_walk = [
        os.path.join(base_dir, "src"),
        os.path.join(base_dir, "tests"),
        os.path.join(base_dir, "docs"),
        base_dir
    ]
    
    seen_files = set()
    
    for d in dirs_to_walk:
        if not os.path.exists(d):
            continue
        if d == base_dir:
            # Just do files in root
            files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
            for file in files:
                if file.endswith((".py", ".md")):
                    filepath = os.path.join(d, file)
                    if filepath not in seen_files:
                        res = analyze_file(filepath, base_dir)
                        results.append(res)
                        seen_files.add(filepath)
        else:
            for root, dirs, files in os.walk(d):
                if "venv" in root or "scratch" in root or ".git" in root or "__pycache__" in root or "chroma_db" in root:
                    continue
                for file in files:
                    if file.endswith(".py") or file.endswith(".md"):
                        filepath = os.path.join(root, file)
                        if filepath not in seen_files:
                            res = analyze_file(filepath, base_dir)
                            results.append(res)
                            seen_files.add(filepath)
                    
    with open(os.path.join(base_dir, "repo_data_full.json"), "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
