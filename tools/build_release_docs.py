"""Build the versioned GitHub Pages layout with correct canonical URLs."""

from __future__ import annotations

import argparse
import logging
import re
import shutil
from pathlib import Path

from check_repository_identity import PAGES_URL
from mkdocs.commands.build import build
from mkdocs.config import load_config


def build_release_docs(version: str, output: Path) -> None:
    if re.fullmatch(r"\d+\.\d+\.\d+", version) is None:
        raise ValueError("Documentation version must be a numeric major.minor.patch value")
    root = Path(__file__).resolve().parents[1]
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite documentation output: {output}")
    output.mkdir(parents=True)
    for configuration, directory in (("mkdocs.yml", version), ("website/mkdocs.yml", "website")):
        config = load_config(
            config_file=str(root / configuration),
            strict=True,
            site_dir=str(output / directory),
            site_url=f"{PAGES_URL}{directory}/",
        )
        config.plugins.on_startup(command="build", dirty=False)
        try:
            build(config)
        finally:
            config.plugins.on_shutdown()
    shutil.copytree(output / version, output / "latest")
    (output / "index.html").write_text(
        '<!doctype html><meta http-equiv="refresh" content="0; url=latest/"><title>VAMOS documentation</title>\n',
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    build_release_docs(args.version, args.output)


if __name__ == "__main__":
    main()
