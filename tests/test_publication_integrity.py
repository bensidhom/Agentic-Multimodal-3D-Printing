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

def test_revised_manuscript_source():
    root=Path(__file__).resolve().parents[1]
    text=(root/'paper/manuscript.md').read_text(encoding='utf-8')
    assert text.count('![Figure ')==8
    assert 'INSERT PERMANENT REPOSITORY' not in text
    assert '(Ben Sidhom n.d.' not in text and '(Ben Sidhom, n.d.' not in text
    assert 'T. Chen & Guestrin, 2016' in text and 'L. Chen et al. (2023)' in text
    assert 'three consecutive' in text.lower() and 'temperature targets' in text
    assert '\\frac{2TP_k}' in text
    assert 'https://github.com/bensidhom/Agentic-Multimodal-3D-Printing' in text
