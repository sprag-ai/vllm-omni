"""Build-time opt-in guard against vLLM's automatic startup/profile reseeding.

Patch only the named function, fail closed if the base no longer exposes it.
The normal runtime behavior is retained unless SPRAG_PRESERVE_RNG_STATE=1.
"""

import ast
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("vllm.utils.torch_utils")
path = Path(spec.origin)
source = path.read_text()
function = next(
    node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "set_random_seed"
)
start = function.body[0].lineno - 1
if isinstance(function.body[0], ast.Expr) and isinstance(function.body[0].value, ast.Constant):
    start = function.body[0].end_lineno
lines = source.splitlines(keepends=True)
lines[start:start] = [
    '    if os.environ.get("SPRAG_PRESERVE_RNG_STATE") == "1":\n',
    "        return\n",
]
result = "".join(lines)
ast.parse(result)
path.write_text(result)
