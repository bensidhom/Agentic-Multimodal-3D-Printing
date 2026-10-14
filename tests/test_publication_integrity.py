import importlib.util
from pathlib import Path

def test_archived_evidence():
    script = Path(__file__).resolve().parents[1] / "scripts/verify_evidence.py"
    spec = importlib.util.spec_from_file_location("verify_evidence", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.verify()
    assert result["ae_prediction_runs_verified"] == 15

def test_figure_links_and_repository_attribution():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    for index in range(1,9):
        path = f"assets/figures/figure_{index:02d}.png"
        assert path in readme and (root / path).is_file()
    assert "https://github.com/bensidhom/MultiAgent_SCADA" in readme
    assert "temporal" in readme.lower() and "schematic" in readme.lower()
