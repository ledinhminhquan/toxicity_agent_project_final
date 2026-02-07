from pathlib import Path
import os
import tempfile

from toxicity_agent.config import load_config


def test_load_config_env_interpolation():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cfg.yaml"
        os.environ["FOO_TEST"] = "bar123"
        p.write_text("x: ${FOO_TEST}\n", encoding="utf-8")
        cfg = load_config(str(p))
        assert cfg["x"] == "bar123"
