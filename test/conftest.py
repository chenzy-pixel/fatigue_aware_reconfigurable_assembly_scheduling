from __future__ import annotations

import pytest

from configs import load_config, project_path
from data import load_instance_pickle
from data.generate_orders import InstanceGenerator, PRESSURE_TYPES


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow generator distribution audits",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: long-running generator or dataset audit",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="use --runslow to run slow audits")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def config():
    return load_config("configs/default.json")


@pytest.fixture(scope="session")
def fixed_instance(config):
    return load_instance_pickle(project_path(config["paths"]["instance_cache"]))


@pytest.fixture(scope="session")
def instance_generator(config, fixed_instance):
    return InstanceGenerator(
        fixed_instance,
        config["generator"],
        config=config,
    )


@pytest.fixture(scope="session")
def pressure_records(instance_generator):
    return {
        pressure_type: instance_generator.generate(
            seed=1000000 + index,
            split="train",
            pressure_type=pressure_type,
        )
        for index, pressure_type in enumerate(PRESSURE_TYPES)
    }
