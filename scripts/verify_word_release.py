"""Structural release checks for the visually reviewed Word manuscript."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M='http://schemas.openxmlformats.org/officeDocument/2006/math'
A='http://schemas.openxmlformats.org/drawingml/2006/main'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
NS={'w':W,'m':M,'a':A,'r':R}

def verify(root=ROOT):
    paper=root/'paper/JIM_revised_manuscript_v3.docx'
    with ZipFile(paper) as archive:
        assert archive.testzip() is None
        doc=ET.fromstring(archive.read('word/document.xml'))
        rels=ET.fromstring(archive.read('word/_rels/document.xml.rels'))
        targets={r.get('Id'):r.get('Target') for r in rels}
        images=doc.findall('.//a:blip',NS)
        for number,blip in enumerate(images,1):
            member='word/'+targets[blip.get('{'+R+'}embed')]
            expected=root/f'assets/figures/figure_{number:02d}.png'
            assert archive.read(member)==expected.read_bytes(),number
        tables=doc.findall('.//w:tbl',NS)
        math=doc.findall('.//m:oMath',NS)
        assert len(images)==8 and len(tables)==8 and len(math)==3
        ps=doc.findall('./w:body/w:p',NS)
        assert len(ps)==232
        body='\n'.join(''.join(p.itertext()) for p in ps[:167])
        assert 'INSERT PERMANENT' not in body
        assert '(Ben Sidhom n.d.' not in body and '(Ben Sidhom, n.d.' not in body
        assert 'an ordinary color image' in body and 'or a small piece of text' in body
        url='https://github.com/bensidhom/Agentic-Multimodal-3D-Printing'
        assert url in body and url in targets.values()
        # Standalone reference titles are italic, not their retrieval dates.
        for p in ps[168:]:
            for run in p.findall('w:r',NS):
                if run.find('w:rPr/w:i',NS) is not None:
                    assert 'Retrieved' not in ''.join(run.itertext())
        base=root/'paper/source_before_revision.docx'
        if base.exists():
            with ZipFile(base) as original:
                old=ET.fromstring(original.read('word/document.xml'))
            for selector in ('.//w:tbl','.//m:oMath'):
                assert [ET.tostring(n) for n in old.findall(selector,NS)]==[ET.tostring(n) for n in doc.findall(selector,NS)]
    return {'docx_sha256':hashlib.sha256(paper.read_bytes()).hexdigest(),
            'figures_byte_identical_to_accepted_assets':8,'tables':8,'native_equations':3,
            'reference_entries':64,'repository_link_verified':True,'structural_checks':'passed'}

if __name__=='__main__':
    print(json.dumps(verify(),indent=2))
