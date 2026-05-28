from pathlib import Path


def test_yuhua_method_packages_point_to_integrated_upstreams():
    import tidal.methods.bslora as bslora
    import tidal.methods.lara as lara

    root = Path(__file__).resolve().parents[1]

    assert bslora.UPSTREAM_REPOSITORY == "https://github.com/yuhua-zhou/BSLoRA"
    assert bslora.UPSTREAM_PATH == "external/BSLoRA"
    assert (root / bslora.UPSTREAM_PATH / "sharelora_fine_tune.py").is_file()
    assert (root / bslora.UPSTREAM_PATH / "peft").is_dir()

    assert lara.UPSTREAM_REPOSITORY == "https://github.com/yuhua-zhou/LaRA"
    assert lara.UPSTREAM_PATH == "external/LaRA"
    assert (root / lara.UPSTREAM_PATH / "train_nas_model.py").is_file()
    assert (root / lara.UPSTREAM_PATH / "model").is_dir()
