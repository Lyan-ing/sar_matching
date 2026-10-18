"""Reconstruct derived summary only; never rerun single-shot predictions."""
import sys,importlib.util
sys.dont_write_bytecode=True
from pathlib import Path
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('new_confirmation',root/'confirm.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.finish_summary();print('confirmation summary reconstructed; no inference repeated')
