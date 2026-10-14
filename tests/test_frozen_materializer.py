import csv
import hashlib
import importlib.util
from pathlib import Path

def module():
    path=Path(__file__).resolve().parents[1]/'scripts/materialize_frozen_vision.py'
    spec=importlib.util.spec_from_file_location('materializer',path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

def test_frozen_image_and_separate_labels_are_preserved(tmp_path):
    images=tmp_path/'images'
    images.mkdir()
    raw=b'fixture image bytes'
    (images/'renamed.jpg').write_bytes(raw)
    manifest=tmp_path/'manifest.csv'
    row={'split':'test','image_path':'data\\external\\frozen.jpg','image_sha256':hashlib.sha256(raw).hexdigest(),
         'classification_target':'underextrusion','normalized_boxes':'1 0.5 0.5 0.2 0.3'}
    with manifest.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    before=manifest.read_bytes()
    result=module().materialize(images,tmp_path/'prepared',manifest)
    assert manifest.read_bytes()==before
    assert Path(result[0]['image_path']).read_bytes()==raw
    assert Path(result[0]['label_path']).read_text()=='1 0.5 0.5 0.2 0.3\n'
    assert Path(result[0]['classification_path']).is_file()

def test_missing_hash_is_rejected_before_materializing(tmp_path):
    import pytest
    images=tmp_path/'images'
    images.mkdir()
    manifest=tmp_path/'manifest.csv'
    manifest.write_text('image_sha256\nmissing\n',encoding='utf-8')
    with pytest.raises(ValueError,match='missing'):
        module().materialize(images,tmp_path/'prepared',manifest)
    assert not (tmp_path/'prepared').exists()
