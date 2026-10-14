"""Report credential/path findings by location, never by secret value."""
import json
import re
from pathlib import Path

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
    for path in ROOT.rglob('*'):
        if not path.is_file() or any(part in {'.git','__pycache__','.venv','artifacts'} for part in path.parts):
            continue
        if path.suffix.lower() not in {'.py','.ps1','.md','.json','.csv','.yaml','.yml','.txt'}:
            continue
        text=path.read_text(encoding='utf-8-sig')
        count+=1
        for kind,pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                findings.append({'path':path.relative_to(ROOT).as_posix(),'line':text.count('\n',0,match.start())+1,'kind':kind})
    report={'text_files_scanned':count,'findings':findings,'sensitive_values_printed':False}
    print(json.dumps(report,indent=2))
    if findings:
        raise SystemExit(1)

if __name__=='__main__':
    audit()
