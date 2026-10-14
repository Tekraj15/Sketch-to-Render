import json
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional

# NOTE: the manifest is JSON Lines on purpose -- .gitignore ignores *.json repo-wide.
STRATA = ("scribble", "lineart", "shaded_drawing", "paper_photo", "real_photo")
OWNERS = ("user", "third_party")
INPUT_MODES = ("Digital Sketch", "Paper Sketch", "Real Photo")


@dataclass
class BenchItem:
    id: str
    stratum: str
    sketch: str                       # path relative to repo root
    prompt: str
    style_dna: str
    domain: str
    owner: str
    reference: Optional[str] = None   # paired photo (enables DINOv2 / KID)
    input_mode: Optional[str] = None  # hint for the import preprocessing path
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class ManifestError(ValueError):
    pass


def validate_item(item: BenchItem, root: str = ".", check_files: bool = True) -> None:
    """Raises ManifestError describing the first problem found."""
    if not item.id or any(c.isspace() for c in item.id):
        raise ManifestError(f"invalid id {item.id!r}")
    if item.stratum not in STRATA:
        raise ManifestError(f"{item.id}: stratum {item.stratum!r} not in {STRATA}")
    if item.owner not in OWNERS:
        raise ManifestError(f"{item.id}: owner {item.owner!r} not in {OWNERS}")
    if item.input_mode is not None and item.input_mode not in INPUT_MODES:
        raise ManifestError(f"{item.id}: input_mode {item.input_mode!r} not in {INPUT_MODES}")
    if not item.prompt.strip():
        raise ManifestError(f"{item.id}: empty prompt")
    if check_files:
        for label, rel in (("sketch", item.sketch), ("reference", item.reference)):
            if rel is not None and not os.path.isfile(os.path.join(root, rel)):
                raise ManifestError(f"{item.id}: {label} file not found: {rel}")


def load_manifest(path: str, root: str = ".", check_files: bool = True) -> List[BenchItem]:
    items: List[BenchItem] = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                item = BenchItem(**json.loads(line))
            except (json.JSONDecodeError, TypeError) as e:
                raise ManifestError(f"{path}:{lineno}: {e}") from e
            if item.id in seen:
                raise ManifestError(f"{path}:{lineno}: duplicate id {item.id!r}")
            validate_item(item, root, check_files)
            seen.add(item.id)
            items.append(item)
    return items


def append_item(path: str, item: BenchItem, root: str = ".") -> None:
    """Validates against the existing manifest, then appends one line."""
    existing = load_manifest(path, root, check_files=False) if os.path.exists(path) else []
    if any(e.id == item.id for e in existing):
        raise ManifestError(f"duplicate id {item.id!r}")
    validate_item(item, root, check_files=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(item.to_json() + "\n")


def select_items(items: List[BenchItem], ids: Optional[List[str]] = None,
                 strata: Optional[List[str]] = None, limit: Optional[int] = None) -> List[BenchItem]:
    out = [i for i in items if (not ids or i.id in ids) and (not strata or i.stratum in strata)]
    if ids:
        missing = set(ids) - {i.id for i in out}
        if missing:
            raise ManifestError(f"unknown item ids: {sorted(missing)}")
    return out[:limit] if limit else out
