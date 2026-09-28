from benchmark_parallel import _choose


def test_recommendation_uses_five_percent_band_and_memory_guard():
    rows = [
        {"status": "ok", "workers": 8, "training_wall_seconds": 109.0,
         "minimum_available_memory_gib": 8.0, "peak_gpu_memory_mib": 2000,
         "gpu_memory_total_mib": 16000},
        {"status": "ok", "workers": 16, "training_wall_seconds": 103.0,
         "minimum_available_memory_gib": 6.0, "peak_gpu_memory_mib": 3500,
         "gpu_memory_total_mib": 16000},
        {"status": "ok", "workers": 32, "training_wall_seconds": 100.0,
         "minimum_available_memory_gib": 3.0, "peak_gpu_memory_mib": 15000,
         "gpu_memory_total_mib": 16000},
    ]
    assert _choose(rows, "training_wall_seconds") == 16
    rows[0]["training_wall_seconds"] = 108.0
    assert _choose(rows, "training_wall_seconds") == 8
