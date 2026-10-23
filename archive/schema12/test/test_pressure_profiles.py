from __future__ import annotations

from data.generate_orders import PRESSURE_TYPES


def test_pressure_profile_configuration_is_complete(config):
    profiles = config["generator"]["pressure_profiles"]
    assert tuple(profiles) == PRESSURE_TYPES
    for profile in profiles.values():
        assert len(profile["release_windows"]) == config["generator"][
            "wave_count"
        ]
        assert profile["order_count"][0] <= profile["order_count"][1]
        assert (
            profile["operations_per_order"][0]
            <= profile["operations_per_order"][1]
        )
        assert 0.60 <= profile["dominant_module_share"][0]
        assert (
            profile["dominant_module_share"][0]
            <= profile["dominant_module_share"][1]
            < 1.0
        )
