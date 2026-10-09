"""
Regression guard for the module code behind the notebooks
==========================================================
The notebooks ship pre-executed, so their stored output is not evidence that the
code behind them still runs. Two defects of 20 September went unnoticed for that
reason (see 9_misc/plan/open-items.md, section 1.1):

- all three copies of tip_tools.py were unimportable (an unclosed docstring);
- ipscore_kit.save_questionnaire was missing, so 5_ipscore/0_questionnaire.ipynb
  could not save.

These tests are the two checks proposed there:

1. Every notebook's local imports load, and every attribute the notebook reads
   from them resolves -- with getattr, against the imported module.
2. The tip_tools.py copies are byte-identical and all compile.

A module that needs a package not installed here (epo.tipdata exists only on
TIP) is skipped with the missing package named, not failed.
"""

import ast
import importlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Earlier worked examples, kept for reference -- not course material.
EXCLUDED_DIRS = {"legacy"}

# The copies tip_tools.py's own header lists.
TIP_TOOLS_COPIES = [
    "4_patentreports/antibiotic_resistance/tip_tools.py",
    "5_ipscore/tip_tools.py",
    "9_misc/ipscore/tip_tools.py",
]


def _course_notebooks():
    notebooks = []
    for path in sorted(REPO_ROOT.rglob("*.ipynb")):
        rel = path.relative_to(REPO_ROOT)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if EXCLUDED_DIRS & set(rel.parts):
            continue
        notebooks.append(path)
    return notebooks


def _code_cells(notebook):
    """Each code cell's source, with IPython magics and shell escapes dropped."""
    with open(notebook, encoding="utf-8") as f:
        cells = json.load(f)["cells"]
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        if isinstance(source, list):
            source = "".join(source)
        lines = [
            line for line in source.splitlines()
            if not line.lstrip().startswith(("%", "!"))
        ]
        yield index, "\n".join(lines)


def _is_local(name, folder):
    top = name.split(".")[0]
    return (folder / f"{top}.py").is_file() or (folder / top / "__init__.py").is_file()


def _local_references(notebook):
    """What a notebook takes from the .py files next to it.

    Returns (modules, names): modules maps each alias bound by ``import x`` /
    ``import x as y`` to its module and the attributes the notebook reads from
    it; names maps each ``from x import a`` module to the names it imports.
    """
    folder = notebook.parent
    aliases = {}
    names = {}
    trees = []
    for index, source in _code_cells(notebook):
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            pytest.fail(f"{notebook.name} cell {index} is not valid Python: {exc}")
        trees.append(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_local(alias.name, folder):
                        aliases[alias.asname or alias.name.split(".")[0]] = (
                            alias.name if alias.asname else alias.name.split(".")[0]
                        )
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                if _is_local(node.module, folder):
                    names.setdefault(node.module, set()).update(
                        a.name for a in node.names if a.name != "*"
                    )
    modules = {alias: (module, set()) for alias, module in aliases.items()}
    for tree in trees:
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.ctx, ast.Load)
                and isinstance(node.value, ast.Name)
                and node.value.id in modules
            ):
                modules[node.value.id][1].add(node.attr)
    return modules, names


@pytest.fixture
def isolated_imports(monkeypatch):
    """Import a notebook's modules the way Jupyter would, then forget them.

    Several folders carry a module of the same name (tip_tools), so each
    notebook must import its own copy. Modules that were already loaded before
    the test (the query-library engine the other tests use) are put back as
    they were, so their classes keep their identity.
    """
    before = dict(sys.modules)

    def enter(folder):
        monkeypatch.chdir(folder)
        monkeypatch.syspath_prepend(str(folder))
        for name, module in list(sys.modules.items()):
            if _is_local(name, folder):
                del sys.modules[name]

    yield enter

    # Forget only the repository's own modules: third-party extension modules
    # (numpy) cannot be loaded a second time in one process.
    for name, module in list(sys.modules.items()):
        if name not in before and _is_from_repo(module):
            del sys.modules[name]
    sys.modules.update(before)


def _is_from_repo(module):
    path = getattr(module, "__file__", None)
    if not path:
        return False
    path = Path(path).resolve()
    return REPO_ROOT in path.parents and not any(
        part.startswith(".") for part in path.relative_to(REPO_ROOT).parts
    )


def _import(module, folder):
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name and not _is_local(exc.name, folder):
            pytest.skip(f"{module} needs {exc.name!r}, which is not installed here")
        raise


@pytest.mark.parametrize(
    "notebook",
    _course_notebooks(),
    ids=lambda p: str(p.relative_to(REPO_ROOT)),
)
def test_notebook_attributes_resolve(notebook, isolated_imports):
    """Every module a notebook imports loads, and has what the notebook uses."""
    folder = notebook.parent
    isolated_imports(folder)
    modules, names = _local_references(notebook)

    missing = []
    for alias, (module_name, attributes) in sorted(modules.items()):
        module = _import(module_name, folder)
        missing += [f"{module_name}.{a}" for a in sorted(attributes) if not hasattr(module, a)]
    for module_name, imported in sorted(names.items()):
        module = _import(module_name, folder)
        for name in sorted(imported):
            if hasattr(module, name):
                continue
            try:  # a submodule of a package: from dpma import fetch
                importlib.import_module(f"{module_name}.{name}")
            except ModuleNotFoundError:
                missing.append(f"{module_name}.{name}")

    assert not missing, f"{notebook.name} uses names its modules do not define: {missing}"


def test_notebooks_are_found():
    """Guard the guard: a path change must not silently empty the list above."""
    assert len(_course_notebooks()) >= 15


@pytest.mark.parametrize("copy", TIP_TOOLS_COPIES)
def test_tip_tools_copy_compiles(copy):
    path = REPO_ROOT / copy
    compile(path.read_bytes(), str(path), "exec")


def test_tip_tools_copies_are_identical():
    contents = {copy: (REPO_ROOT / copy).read_bytes() for copy in TIP_TOOLS_COPIES}
    reference = contents[TIP_TOOLS_COPIES[0]]
    differing = [copy for copy, data in contents.items() if data != reference]
    assert not differing, (
        f"tip_tools.py copies differ from {TIP_TOOLS_COPIES[0]}: {differing} "
        "-- change one, change all three"
    )


def test_every_tip_tools_copy_is_listed():
    """A new copy has to join the list, or it escapes the identity check."""
    found = sorted(
        str(p.relative_to(REPO_ROOT))
        for p in REPO_ROOT.rglob("tip_tools.py")
        if not any(part.startswith(".") for part in p.relative_to(REPO_ROOT).parts)
        and not EXCLUDED_DIRS & set(p.relative_to(REPO_ROOT).parts)
    )
    assert found == sorted(TIP_TOOLS_COPIES)
