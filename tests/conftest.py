"""
Shared test helper.

Every tool has a module named `handler.py`, so a plain `sys.path.insert` +
`import handler` makes the first test file to run win the `handler` entry in
sys.modules -- later files then silently get the wrong module. Load each tool
under its own module name instead.
"""
import importlib.util
import pathlib
import sys

TOOLS = pathlib.Path(__file__).resolve().parent.parent / "tools"


def load_tool(tool_name: str):
    """Import <repo>/tools/<tool_name>/handler.py as a uniquely-named module."""
    path = TOOLS / tool_name / "handler.py"
    mod_name = f"_tool_{tool_name}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module
