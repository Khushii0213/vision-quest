"""Every third-party import in the project must be declared in requirements.txt."""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMPORT_TO_DIST = {"PIL": "pillow", "streamlit": "streamlit", "numpy": "numpy", "torch": "torch", "torchvision": "torchvision"}


def declared():
    names = set()
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            names.add(re.split(r"[<>=!~\\[ ;]", line)[0].lower())
    return names


def third_party_imports():
    found = set()
    files = [ROOT / "app.py", *(ROOT / "src").rglob("*.py"), *(ROOT / "scripts").glob("*.py")]
    for f in files:
        for node in ast.walk(ast.parse(f.read_text())):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module]
            for m in mods:
                top = m.split(".")[0]
                if top not in sys.stdlib_module_names and top not in {"src", "app"}:
                    found.add(top)
    return found


def test_all_imports_declared():
    missing = {m for m in third_party_imports() if IMPORT_TO_DIST.get(m, m).lower() not in declared()}
    assert not missing, f"Imports missing from requirements.txt: {missing}"
