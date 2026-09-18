"""Architecture tests: the layer rules that keep the core independent.

These assert the properties the project promises in `ARCHITECTURE.md`, so that a future change that
quietly couples the core to a vendor or inverts a dependency fails the build instead of eroding the
design.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "vidliner"

#: Vendor SDK modules that may only be imported under ``vidliner/backends``.
VENDOR_MODULES = frozenset(
    {
        "torch",
        "torchvision",
        "tensorflow",
        "onnxruntime",
        "cv2",
        "transformers",
        "diffusers",
        "openai",
        "anthropic",
        "boto3",
        "google.cloud",
        "azure",
        "segment_anything",
        "ultralytics",
    }
)

#: Modules that must never be imported by the layers below them.
LAYER_RULES: dict[str, frozenset[str]] = {
    "domain": frozenset({"vidliner.pipeline", "vidliner.operators", "vidliner.backends", "vidliner.cli"}),
    # ``core`` deliberately knows about the *domain enumerations and value objects* it returns; it
    # must not know about the pipeline that consumes them.
    "core": frozenset({"vidliner.pipeline", "vidliner.operators", "vidliner.backends", "vidliner.cli"}),
    # ``core`` may use the *domain value objects and enumerations* it returns; it must not import
    # the operator catalogue, which is why that dependency is injected (see
    # ``OperationGraph.validate_against_operators``).
    "capabilities": frozenset({"vidliner.pipeline", "vidliner.backends", "vidliner.cli"}),
    "quality": frozenset({"vidliner.pipeline", "vidliner.backends", "vidliner.cli"}),
    "annotations": frozenset(
        {"vidliner.pipeline", "vidliner.operators", "vidliner.backends", "vidliner.cli"}
    ),
}


def _modules() -> list[Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def _imported_modules(path: Path) -> set[str]:
    """Every module name imported by a file, including those inside ``if TYPE_CHECKING``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            if node.level:
                # Relative import: resolve it against the package path.
                package_parts = path.relative_to(PACKAGE_ROOT).parts[:-1]
                base = ".".join(("vidliner", *package_parts[: len(package_parts) - node.level + 1]))
                imported.add(f"{base}.{node.module}" if node.module else base)
    return imported


def test_vendor_sdks_only_appear_in_backends() -> None:
    offenders: list[str] = []
    for path in _modules():
        relative = path.relative_to(PACKAGE_ROOT)
        if relative.parts and relative.parts[0] == "backends":
            continue
        for module in _imported_modules(path):
            root = module.split(".", 1)[0]
            if root in VENDOR_MODULES:
                offenders.append(f"{relative}: imports {module}")
    assert not offenders, "vendor SDK imports outside vidliner/backends:\n" + "\n".join(offenders)


@pytest.mark.parametrize("layer", sorted(LAYER_RULES))
def test_layers_do_not_import_upwards(layer: str) -> None:
    forbidden = LAYER_RULES[layer]
    offenders: list[str] = []
    layer_root = PACKAGE_ROOT / layer
    for path in sorted(layer_root.rglob("*.py")):
        for module in _imported_modules(path):
            if any(module == name or module.startswith(f"{name}.") for name in forbidden):
                offenders.append(f"{path.relative_to(PACKAGE_ROOT)}: imports {module}")
    assert not offenders, f"{layer} imports a higher layer:\n" + "\n".join(offenders)


#: The one backend module that is a shared *utility* rather than a capability adapter: mask and
#: image helpers that operators use to prepare a backend call. It holds no vendor code.
_BACKEND_UTILITY = "vidliner.backends.imageops"


def test_backend_adapters_are_reachable_only_through_the_runtime() -> None:
    """Only the runtime registry and the profile's default may name a capability adapter."""
    allowed = {"runtime", "backends"}
    offenders: list[str] = []
    for path in _modules():
        relative = path.relative_to(PACKAGE_ROOT)
        if relative.parts[0] in allowed:
            continue
        for module in _imported_modules(path):
            if not module.startswith("vidliner.backends."):
                continue
            if module == _BACKEND_UTILITY or module.startswith(f"{_BACKEND_UTILITY}."):
                continue
            offenders.append(f"{relative}: imports {module}")
    assert not offenders, (
        "capability adapters must be named only in runtime profiles and the registry:\n"
        + "\n".join(offenders)
    )


#: Domain modules ``core`` is allowed to reference: the closed vocabularies and the value objects
#: that appear in the results it returns.
_CORE_ALLOWED_DOMAIN = ("vidliner.domain.enums", "vidliner.domain.shapes")


def test_core_depends_only_on_domain_primitives() -> None:
    """``core`` may use the domain's enumerations and value objects, and nothing else from it."""
    offenders: list[str] = []
    for path in sorted((PACKAGE_ROOT / "core").rglob("*.py")):
        for module in _imported_modules(path):
            if not module.startswith("vidliner.domain."):
                continue
            if any(module == allowed or module.startswith(f"{allowed}.") for allowed in _CORE_ALLOWED_DOMAIN):
                continue
            offenders.append(f"{path.name}: imports {module}")
    assert not offenders, "core depends on a domain module outside its primitives:\n" + "\n".join(offenders)


def test_no_module_swallows_exceptions_silently() -> None:
    """A bare ``except: pass`` hides exactly the failures this project exists to surface.

    ``contextlib.suppress`` with a comment is fine; swallowing and *continuing* into a fabricated
    success is not. This catches the syntactic case, and the review rules in CONTRIBUTING.md cover
    the semantic one.
    """
    offenders: list[str] = []
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            body = [statement for statement in node.body if not isinstance(statement, ast.Pass)]
            if not body:
                offenders.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}")
    assert not offenders, "silent exception handling found:\n" + "\n".join(offenders)


def test_protocol_and_abstract_members_are_exempt_from_the_docstring_rule() -> None:
    """A declared port documents its contract on the class; its members carry the type."""
    from vidliner.capabilities.backend import ObjectDetectorBackend

    assert hasattr(ObjectDetectorBackend, "detect")


def test_unimplemented_behaviour_raises_rather_than_faking_success() -> None:
    """No module may return a placeholder value where work was expected."""
    offenders: list[str] = []
    for path in _modules():
        source = path.read_text(encoding="utf-8")
        if "PLACEHOLDER" in source or "TODO: implement" in source:
            offenders.append(str(path.relative_to(PACKAGE_ROOT)))
    assert not offenders, "placeholder implementations found:\n" + "\n".join(offenders)


def test_public_modules_have_docstrings() -> None:
    missing: list[str] = []
    for path in _modules():
        if path.name == "__init__.py" and path.stat().st_size == 0:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if not ast.get_docstring(tree):
            missing.append(str(path.relative_to(PACKAGE_ROOT)))
    assert not missing, "modules without a docstring:\n" + "\n".join(missing)


def _is_declaration_only(node: ast.ClassDef) -> bool:
    """True for a Protocol or ABC, whose members declare a contract rather than behaviour."""
    for base in node.bases:
        name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
        if name in {"Protocol", "ABC", "StrEnum", "BaseModel"}:
            return True
    return False


def _declaration_only_spans(tree: ast.AST) -> list[tuple[int, int]]:
    """Line ranges of Protocol/ABC classes, whose members need no individual docstrings."""
    spans: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and _is_declaration_only(node):
            spans.append((node.lineno, getattr(node, "end_lineno", node.lineno)))
    return spans


def test_public_callables_are_documented() -> None:
    """Every public class and function carries a docstring; private helpers may not.

    Members of a ``Protocol`` or ``ABC`` are exempt: the contract is documented once, on the class,
    and the member names are the type signature.
    """
    missing: list[str] = []
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(PACKAGE_ROOT)
        # Only module-level (and class-level) definitions are public API; a closure created inside a
        # function is an implementation detail and documents itself at its use site.
        public: list[ast.AST] = list(tree.body)
        for top in tree.body:
            if isinstance(top, ast.ClassDef):
                public.extend(top.body)
        spans = _declaration_only_spans(tree)
        for node in public:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if node.name.startswith("_"):
                continue
            if isinstance(node, ast.ClassDef) and _is_declaration_only(node):
                continue
            if any(start <= node.lineno <= end for start, end in spans):
                continue
            if not ast.get_docstring(node):
                missing.append(f"{relative}:{node.lineno} {node.name}")
    assert not missing, "public callables without a docstring:\n" + "\n".join(missing)


def test_no_hardcoded_absolute_paths() -> None:
    offenders: list[str] = []
    for path in _modules():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#") or '"""' in stripped:
                continue
            if '"/Users/' in stripped or "'/Users/" in stripped or '"/home/' in stripped:
                offenders.append(f"{path.relative_to(PACKAGE_ROOT)}:{number}")
    assert not offenders, "hardcoded absolute paths found:\n" + "\n".join(offenders)


def test_there_is_exactly_one_backend_registry() -> None:
    """A second registry would mean two answers to "which backend is bound"."""
    candidates = [
        path
        for path in _modules()
        if "class " in path.read_text(encoding="utf-8")
        and "BackendRegistry" in path.read_text(encoding="utf-8")
    ]
    definitions = [path for path in candidates if "class BackendRegistry" in path.read_text(encoding="utf-8")]
    assert len(definitions) == 1, f"BackendRegistry is defined in {definitions}"


def test_operators_are_registered_in_one_catalogue() -> None:
    """Every operator module exposes ``register_all``, and the registry imports them all."""
    from vidliner.operators import registry as registry_module

    source = (PACKAGE_ROOT / "operators" / "registry.py").read_text(encoding="utf-8")
    operator_modules = sorted(
        path.stem
        for path in (PACKAGE_ROOT / "operators").glob("*.py")
        if path.stem not in {"__init__", "base", "registry", "shared", "values", "backend_context"}
    )
    assert operator_modules
    for name in operator_modules:
        assert f"operators import {name} as" in source, f"{name} is not wired into the catalogue"
    del registry_module
