"""Reusable scheduling regression cases."""
from collections import Counter

import csv

import json

import math

from pathlib import Path

import torch

from configs import project_path

from configs.config import PROJECT_ROOT

from configs.normalization import apply_normalization_manifest

from configs.runtime import attach_runtime_manifest

from configs.formal_preferences import formal_preferences

from data.dataset import load_dataset_split

from data.distribution import protocol_hashes, training_sampling_plan

from data.selection import subset_snapshot

from result.io import write_json

from scripts.prepare_data_v2 import verify_legacy_files

def load_run_config(path):

    config = json.loads(Path(path).read_text(encoding="utf-8"))

    saved_runtime_manifest = config.pop("runtime_manifest", None)

    apply_normalization_manifest(config, project_root=PROJECT_ROOT)

    attach_runtime_manifest(config)

    if saved_runtime_manifest != config["runtime_manifest"]:

        raise ValueError("saved run runtime manifest differs from the current implementation")

    return config
