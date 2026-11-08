"""Small local Python import closure; parses source without importing a policy."""
from __future__ import annotations
import ast
from pathlib import Path


def runtime_files(source, entry='policy.py'):
    root = Path(source).resolve(); pending = [entry]; found = set()
    def add_module(name):
        parts = name.split('.') if name else []
        candidates = [Path(*parts).with_suffix('.py'), Path(*parts) / '__init__.py'] if parts else []
        for path in candidates:
            if (root / path).is_file():
                pending.append(path.as_posix())
                for parent in path.parents:
                    init = parent / '__init__.py'
                    if (root / init).is_file(): pending.append(init.as_posix())
                return
    while pending:
        name = pending.pop(); relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or relative.suffix != '.py':
            raise ValueError(f'Invalid local runtime dependency: {name}')
        name = relative.as_posix()
        if name in found: continue
        path = root / relative
        if not path.is_file(): raise FileNotFoundError(path)
        found.add(name); tree = ast.parse(path.read_text())
        package = list(relative.parent.parts)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'RUNTIME_DEPENDENCIES' for t in node.targets):
                # A nested frozen parent may declare paths within its own
                # namespace. Resolve from the declaring package outward; a
                # root-relative declaration still resolves at the root.
                for declared in ast.literal_eval(node.value):
                    dep=Path(declared)
                    if dep.is_absolute() or '..' in dep.parts or dep.suffix!='.py':
                        raise ValueError(f'Invalid local runtime dependency: {declared}')
                    bases=[relative.parent,*relative.parent.parents]
                    matched=next((base/dep for base in bases if (root/base/dep).is_file()),None)
                    if matched is None:raise FileNotFoundError(f'{path}: {declared}')
                    pending.append(matched.as_posix())
            elif isinstance(node, ast.Import):
                for alias in node.names: add_module(alias.name)
            elif isinstance(node, ast.ImportFrom):
                base = package[:len(package) - node.level + 1] if node.level else []
                if node.module: base += node.module.split('.')
                add_module('.'.join(base))
                for alias in node.names:
                    if alias.name != '*': add_module('.'.join(base + [alias.name]))
            elif isinstance(node, ast.Call):
                fn = node.func
                dynamic = isinstance(fn, ast.Name) and fn.id == '__import__' or isinstance(fn, ast.Attribute) and fn.attr == 'import_module'
                if dynamic and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    add_module(node.args[0].value)
    return sorted(found)
