"""Export the revised Word body, tables, figures and equations for GitHub readers."""
from pathlib import Path
from zipfile import ZipFile
from lxml import etree

ROOT=Path(__file__).resolve().parents[1]
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'm':'http://schemas.openxmlformats.org/officeDocument/2006/math',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
W='{'+NS['w']+'}'
M='{'+NS['m']+'}'
R='{'+NS['r']+'}'
EQUATIONS={102:[r'$c_k(x)$',r'$\max_k c_k(x)$'],
           111:[r'$F1_k = \frac{2TP_k}{2TP_k + FP_k + FN_k}$']}

def visible(node):
    return ''.join(node.xpath('.//w:t/text() | .//m:t/text()',namespaces=NS))

def export(source=None):
    source=source or ROOT/'paper/JIM_revised_manuscript_v3.docx'
    with ZipFile(source) as archive:
        doc=etree.fromstring(archive.read('word/document.xml'))
        rels=etree.fromstring(archive.read('word/_rels/document.xml.rels'))
    targets={r.get('Id'):r.get('Target') for r in rels}
    lines=[]
    p_index=-1
    figure=0
    tables=0
    for node in doc.find(W+'body'):
        if node.tag==W+'p':
            p_index+=1
            if node.xpath('.//w:drawing',namespaces=NS):
                figure+=1
                lines.append(f'![Figure {figure}](../assets/figures/figure_{figure:02d}.png)')
                continue
            text=[]
            math_index=0
            for child in node:
                if child.tag==M+'oMath':
                    text.append(EQUATIONS[p_index][math_index])
                    math_index+=1
                elif child.tag==W+'hyperlink':
                    text.append(f'[{visible(child)}]({targets[child.get(R+"id")]})')
                elif child.tag in {W+'r',W+'fldSimple'}:
                    part=visible(child)
                    if part and child.find('./'+W+'rPr/'+W+'i') is not None:
                        part='*'+part+'*'
                    text.append(part)
            text=''.join(text).strip()
            if not text:
                continue
            style=node.xpath('./w:pPr/w:pStyle/@w:val',namespaces=NS)
            style=style[0] if style else 'Normal'
            prefix={'Title':'# ','Heading1':'## ','Heading2':'### '}.get(style,'')
            lines.append(prefix+text)
        elif node.tag==W+'tbl':
            tables+=1
            matrix=[]
            for row in node.findall(W+'tr'):
                matrix.append([visible(cell).replace('|','\\|').replace('\n','<br>') for cell in row.findall(W+'tc')])
            lines.append('\n'.join(['| '+' | '.join(matrix[0])+' |',
                         '| '+' | '.join('---' for _ in matrix[0])+' |',
                         *['| '+' | '.join(row)+' |' for row in matrix[1:]]]))
    assert figure==8 and tables==8
    out=ROOT/'paper/manuscript.md'
    out.write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Full manuscript Markdown exported: {figure} figures and {tables} tables')

if __name__=='__main__':
    export()
