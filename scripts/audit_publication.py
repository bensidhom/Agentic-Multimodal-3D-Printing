"""Report credential/path findings by location, never by secret value."""
import json
import re
import subprocess
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
PATTERNS={
    'github_token':re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b'),
    'openai_key':re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}\b'),
    'private_key':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'credential_in_url':re.compile(r'https?://[^\s/:]+:[^\s/@]+@'),
    'signed_link':re.compile(r'https?://\S+[?&](?:X-Amz-Signature|api_key|token)=[^\s&]+',re.I),
    'private_user_path':re.compile(r'[A-Za-z]:[/\\]+Users[/\\]+[^\s\"\']+'),
}

def audit():
    findings=[]
    count=0
    public=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    for relative in sorted(set(public)):
        path=ROOT/relative
        if not path.is_file() or any(part in {'.git','__pycache__','.venv','artifacts'} for part in path.parts):
            continue
        if path.suffix.lower()=='.docx':
            with ZipFile(path) as archive:
                parts=[(name,archive.read(name).decode('utf-8-sig')) for name in archive.namelist() if name.endswith(('.xml','.rels'))]
        elif path.suffix.lower() in {'.py','.ps1','.md','.json','.csv','.tsv','.gcode','.yaml','.yml','.txt'}:
            parts=[('',path.read_text(encoding='utf-8-sig'))]
        else:
            continue
        for part,text in parts:
            count+=1
            for kind,pattern in PATTERNS.items():
                for match in pattern.finditer(text):
                    findings.append({'path':path.relative_to(ROOT).as_posix(),'part':part,'line':text.count('\n',0,match.start())+1,'kind':kind})
    report={'text_files_scanned':count,'findings':findings,'sensitive_values_printed':False}
    print(json.dumps(report,indent=2))
    if findings:
        raise SystemExit(1)

if __name__=='__main__':
    audit()
