"""
CLI argument parsing and validation helpers.
"""

from __future__ import annotations

import argparse

from vamos.foundation.core.experiment_config import ExperimentConfig

from .args import build_parser, build_pre_parser
from .common import collect_nsgaii_variation_args, _collect_generic_variation
from .loaders import load_spec_defaults
from .spec_validation import validate_experiment_spec
from .validation import finalize_args


# Config-file aware parser.
def parse_args(default_config: ExperimentConfig) -> argparse.Namespace:
    pre_parser = build_pre_parser()
    pre_args, remaining = pre_parser.parse_known_args()

    if getattr(pre_args, "validate_config", False) and not getattr(pre_args, "config", None):
        pre_parser.error("--validate-config requires --config.")

    try:
        spec_defaults = load_spec_defaults(pre_args.config)
    except Exception as exc:
        pre_parser.error(str(exc))
    parser = build_parser(
        default_config=default_config,
        pre_parser=pre_parser,
        spec_defaults=spec_defaults,
    )
    if pre_args.config:
        try:
            validate_experiment_spec(spec_defaults.spec, parser=parser)
        except Exception as exc:
            pre_parser.error(str(exc))
        if getattr(pre_args, "validate_config", False):
            parser.exit(0, "Config OK.\n")
    args = parser.parse_args(remaining)
    return finalize_args(
        parser,
        args,
        spec_defaults=spec_defaults,
        config_path=getattr(pre_args, "config", None),
    )


__all__ = ["parse_args", "collect_nsgaii_variation_args", "_collect_generic_variation"]
