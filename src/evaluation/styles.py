import ast
from typing import Dict


def load_style_table(app_path: str = "app.py") -> Dict[str, str]:
    """
    Reads the live Style-DNA table straight from app.py so the harness never
    keeps its own copy. app.py defines the table inline in process_sketch()
    (and currently assigns `styles` twice), so we take the LAST literal
    `styles = {...}` assignment inside that function -- the one the app uses.
    Switch this to a plain import once the table is extracted into a module.
    """
    with open(app_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=app_path)

    table = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "process_sketch":
            for stmt in ast.walk(node):
                if (isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Dict)
                        and any(isinstance(t, ast.Name) and t.id == "styles" for t in stmt.targets)):
                    if table is None or stmt.lineno > table[0]:
                        table = (stmt.lineno, ast.literal_eval(stmt.value))
    if table is None:
        raise ValueError(f"No literal `styles = {{...}}` found in process_sketch() of {app_path}")
    return table[1]


def compose_prompt(user_prompt: str, style_suffix: str) -> str:
    # Mirrors app.py's composition: f"{prompt}, {style_suffix}".strip(", ")
    return f"{user_prompt}, {style_suffix}".strip(", ")
