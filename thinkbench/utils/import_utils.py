import importlib
import os
from typing import Optional, Sequence
import re

from constants import LIBRARY_ROOT


def import_modules_from_folder(
    folder_name: str, extra_roots: Optional[Sequence[str]] = None
) -> None:
    if not LIBRARY_ROOT.joinpath(folder_name).exists():
        raise ValueError(f"{folder_name} doesn't exist in the public library root directory.")

    base_dirs = ["."]
    if extra_roots is not None:
        base_dirs += sorted(extra_roots)
    for base_dir in base_dirs:
        if base_dir.startswith("thinkbench/") and folder_name.startswith("thinkbench/"):
            base_dir = os.path.join(base_dir, re.sub("^thinkbench/", "", folder_name))
        else:
            base_dir = os.path.join(base_dir, folder_name)
        for path in sorted(LIBRARY_ROOT.glob(os.path.join(base_dir, "**/*.py"))):
            filename = path.name
            if filename[0] not in (".", "_"):
                module_name = str(
                    path.relative_to(LIBRARY_ROOT).with_suffix("")
                ).replace(os.sep, ".")
                importlib.import_module(module_name)
