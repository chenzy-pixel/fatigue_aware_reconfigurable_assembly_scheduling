"""Reusable scheduling regression cases."""
from __future__ import annotations

import argparse

from copy import deepcopy

from dataclasses import replace

import hashlib

import json

from pathlib import Path

import sys

import numpy as np

from agent.baselines import HeuristicPolicy, RandomPolicy

from configs import load_config

from data.dataset import load_generated_record

from data.models import load_instance_yaml

from environment import AssemblySchedulingEnv

from support.observation_cases import observation_sha256

ACTUAL_RELATIONS = {("operation", "processing_on", "machine"), ("machine", "served_by", "worker")}

def legacy_projection(obs):

    return replace(obs, relations={key: store for key, store in obs.relations.items() if key not in ACTUAL_RELATIONS})
