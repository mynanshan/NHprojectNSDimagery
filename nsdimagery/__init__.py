"""Small, dependency-light helpers for exploring the NSD-Imagery release."""

from .io import (
    RUN_SPECS,
    build_run_table,
    describe_hdf5,
    extract_masked_betas,
    find_data_root,
    infer_beta_layout,
    load_behavior,
    load_roi,
    read_ctab,
    validate_download,
)

__all__ = [
    "RUN_SPECS",
    "build_run_table",
    "describe_hdf5",
    "extract_masked_betas",
    "find_data_root",
    "infer_beta_layout",
    "load_behavior",
    "load_roi",
    "read_ctab",
    "validate_download",
]
