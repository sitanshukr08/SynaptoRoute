import json
import os

def generate_repo_map(data):
    lines = ["# Repository Map\n"]
    lines.append("## Overview\n")
    lines.append("This document maps the files, their purposes, dependencies, and exported components within the repository.\n")
    
    # Categorize files based on keywords
    entrypoints = []
    apis = []
    clis = []
    dbs = []
    agents = []
    routers = []
    middlewares = []
    configs = []
    background = []
    tests = []
    docs = []
    others = []

    for item in data:
        path = item["path"].replace("\\", "/")
        if path.startswith("tests/"):
            tests.append(item)
            continue
        if path.startswith("docs/") or path.endswith(".md"):
            docs.append(item)
            continue
            
        if "main.py" in path or "entrypoint" in path:
            if "cli" in path:
                clis.append(item)
            else:
                entrypoints.append(item)
        elif "api" in path:
            if "route" in path:
                routers.append(item)
            elif "auth" in path or "middleware" in path:
                middlewares.append(item)
            else:
                apis.append(item)
        elif "db" in path or "memory" in path or "persistence" in path or "vector" in path or "model" in path:
            if "mcp" in path or "job" in path:
                background.append(item)
            else:
                dbs.append(item)
        elif "agent" in path or "workflow" in path or "pipeline" in path or "graph" in path or "rag" in path or "core" in path or "llm" in path:
            agents.append(item)
        elif "config" in path or "settings" in path:
            configs.append(item)
        elif "cli" in path:
            clis.append(item)
        elif "route" in path or "router" in path:
            routers.append(item)
        else:
            others.append(item)

    categories = [
        ("Application Entrypoints", entrypoints),
        ("APIs", apis),
        ("CLI Commands", clis),
        ("Databases & Models", dbs),
        ("Agent Workflows", agents),
        ("Routers", routers),
        ("Middleware", middlewares),
        ("Config Files", configs),
        ("Background Jobs", background),
        ("Documentation", docs),
        ("Tests", tests),
        ("Other Components", others)
    ]

    for cat_name, cat_items in categories:
        if not cat_items: continue
        lines.append(f"### {cat_name}")
        for item in cat_items:
            path = item["path"].replace("\\", "/")
            lines.append(f"- **{path}**")
            # Replace newlines in purpose with space
            purpose = item['purpose'].replace('\n', ' ')
            lines.append(f"  - **Purpose**: {purpose}")
            if path.endswith(".py"):
                lines.append(f"  - **Imports**: {', '.join(item['imports']) if item['imports'] else 'None'}")
                lines.append(f"  - **Exports**: {', '.join(item['exports']) if item['exports'] else 'None'}")
            lines.append("")

    return "\n".join(lines)

def detect_risks(data):
    risks = []
    
    # dependencies mappings
    dep_graph = {}
    for item in data:
        if not item["path"].endswith(".py"):
            continue
        path = item["path"].replace("\\", "/")
        # Convert path to module name assuming src/ is root or base is root
        mod_name = path.replace("src/", "").replace(".py", "").replace("/", ".")
        deps = []
        for imp in item["imports"]:
            if "synaptoroute" in imp or not imp.startswith(("os", "sys", "json", "typing", "ast", "pytest", "langchain")):
                deps.append(imp)
        dep_graph[mod_name] = deps

    for node, edges in dep_graph.items():
        for edge in edges:
            if edge in dep_graph and node in dep_graph[edge]:
                risks.append(f"Circular dependency detected between `{node}` and `{edge}`")

    # Unreachable components check (naive: check if any module imports this module)
    # This is rough and might be inaccurate for dynamically imported or entrypoint files
    imported_modules = set()
    for edges in dep_graph.values():
        for edge in edges:
            imported_modules.add(edge)
            
    unreachable = []
    for node in dep_graph.keys():
        if node not in imported_modules and not ("main" in node or "test" in node or "init" in node or "cli" in node or "app" in node):
            unreachable.append(node)

    if unreachable:
        risks.append(f"Potentially unreachable components (no explicit imports found): {', '.join(unreachable)}")
        
    return risks, dep_graph

def generate_arch_report(data):
    risks, dep_graph = detect_risks(data)
    
    lines = ["# Architecture Report\n"]
    
    lines.append("## Dependency Graph")
    lines.append("```mermaid")
    lines.append("graph TD")
    added = set()
    for node, edges in dep_graph.items():
        # Only include edges that are within the project or internal modules
        for edge in edges:
            if "synaptoroute" in edge or edge in dep_graph:
                clean_node = node.replace(".", "_")
                clean_edge = edge.replace(".", "_")
                line = f"    {clean_node} --> {clean_edge}"
                if line not in added:
                    lines.append(line)
                    added.add(line)
    lines.append("```\n")
    
    lines.append("## Request Flow Graph (Heuristic)")
    lines.append("```mermaid")
    lines.append("graph LR")
    lines.append("    User --> Entrypoint")
    lines.append("    Entrypoint --> Router")
    lines.append("    Router --> Agent")
    lines.append("    Agent --> Tools")
    lines.append("    Agent --> VectorDB")
    lines.append("```\n")

    lines.append("## Agent Interaction Graph (Heuristic)")
    lines.append("```mermaid")
    lines.append("graph TD")
    lines.append("    User_Query --> Routing_Layer")
    lines.append("    Routing_Layer --> RAG_Agent")
    lines.append("    Routing_Layer --> SQL_Agent")
    lines.append("    Routing_Layer --> Fallback")
    lines.append("```\n")

    lines.append("## Execution Graph (Heuristic)")
    lines.append("```mermaid")
    lines.append("graph TD")
    lines.append("    Init --> Config_Load")
    lines.append("    Config_Load --> Main_Loop")
    lines.append("    Main_Loop --> Tool_Execution")
    lines.append("    Tool_Execution --> Main_Loop")
    lines.append("```\n")

    lines.append("## Architectural Risks & Hidden Coupling\n")
    if risks:
        for r in risks:
            lines.append(f"- {r}")
    else:
        lines.append("- No obvious circular dependencies detected.")
    
    lines.append("- **Hidden Coupling**: Check for global state usages in configuration or metrics, which could couple components implicitly.")
    
    return "\n".join(lines)

def main():
    base_dir = r"c:\Users\sitan\OneDrive\Desktop\synaptoroute"
    with open(os.path.join(base_dir, "repo_data_full.json"), "r", encoding='utf-8') as f:
        data = json.load(f)
    
    repo_map = generate_repo_map(data)
    with open(os.path.join(base_dir, "REPOSITORY_MAP.md"), "w", encoding='utf-8') as f:
        f.write(repo_map)

    arch_report = generate_arch_report(data)
    with open(os.path.join(base_dir, "ARCHITECTURE_REPORT.md"), "w", encoding='utf-8') as f:
        f.write(arch_report)

if __name__ == "__main__":
    main()
