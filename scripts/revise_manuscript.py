"""Apply the Oct13 paragraph review while preserving figure, table and math XML."""
import argparse
from copy import deepcopy
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZipFile
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/bensidhom/Agentic-Multimodal-3D-Printing"
NS = {"w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main",
      "m":"http://schemas.openxmlformats.org/officeDocument/2006/math",
      "r":"http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
W = "{" + NS["w"] + "}"
R = "{" + NS["r"] + "}"
XMLSPACE = "{http://www.w3.org/XML/1998/namespace}space"

FULL = {
0: "Local language model agents for monitoring and supervising fused filament 3D printing",
3: "Printer operators need to know what is happening during a print and what action to take. We present a local artificial intelligence (AI) agent system for fused filament fabrication, a form of 3D printing. A supervisor sends data questions to a database agent and printer requests to a printing agent. Tools return records or confirm that a request was accepted. Separate camera and contact acoustic emission (AE) services can request a pause without waiting for the AI conversation. The system keeps agents, tools, stored data and computing tasks separate. An AMD Ryzen AI Max+ 395 computer with 64 GB memory is the designated language-model deployment host. NVIDIA hardware supports visual-model training. We describe labeling, training, storage and tool calls, and an architecture linking records by job, time and commanded position. Supporting tests compare six visual models and five AE classifiers, each fitted with three random seeds on fixed data splits. YOLO11n detection reaches image macro F1 of 0.940 ± 0.043, which combines precision and recall with equal class weighting. Its box average precision at an intersection-over-union threshold of 0.50 (AP50) is 0.720 ± 0.013. A small AE multilayer perceptron reaches macro F1 of 0.773 ± 0.000. These scores support recognition-model selection. They do not measure agent-task success or paired-sensor fusion performance.",
10: "AE model choice also depends on accuracy and computing cost. Linear models, groups of decision trees and small neural networks can all use features taken from an acoustic burst. Their decision thresholds, input scaling and speed affect the monitoring service. A decision threshold is the score needed to choose a class. A model may rank events well but give weaker normal-or-defective decisions at one threshold. We therefore check more than overall accuracy. We also check performance across classes, missed defects, false alarms and the quality of probability estimates.",
14: "2.1 Language interfaces and evaluation",
50: "Tool inputs, outputs and effects",
69: "3.6 Visual defect regions and acoustic event association",
82: "4.1 Data and evaluation splits",
115: "4.6 Agent evaluation protocol",
132: "5.2 Acoustic classification and probability checks",
164: "The benchmark code, archived AE feature data, frozen split lists, model settings, saved predictions, training logs, figures and reproduction instructions are available in the consolidated evidence repository: [[LINK:Consolidated evidence repository]]. Raw external images are obtained through the named dataset provider under its access and reuse conditions; large visual checkpoints are not bundled. The repository records software versions, provenance and reuse conditions. The original system and acquisition implementations remain available in [[LINK:MultiAgent SCADA]], [[LINK:AI-driven analytics and digital twins]] and [[LINK:Acoustic emission monitoring at the edge]].",
}

EDITS = {
6: [("heated plastic", "heated thermoplastic")],
8: [("repeated detections", "consecutive detections"), ("We can then test each task at the speed it needs.", "We can then evaluate sensing, reasoning and operator interaction separately.")],
9: [("also draws boxes", "also predicts bounding boxes")],
11: [("The agent workflow and the recognition tests belong in the same paper, but they need different measures of success.", "The recognition models supply evidence to the agents, so the components and agent tasks require separate evaluation.")],
15: [("agent teamwork", "agent coordination"), ("written production requests", "natural-language operator requests")],
17: [("chance estimate", "probability estimate"), ("choose decision cutoffs", "choose decision thresholds")],
21: [("normal color image", "ordinary color image")],
31: [("the AMD computer that runs the language model from the NVIDIA computer used to develop visual models", "the designated AMD deployment host from the NVIDIA visual-training host")],
32: [("16 CPU cores", "16 central processing unit (CPU) cores"), ("This AMD computer hosts the LLM.", "The AMD computer is the designated local LLM deployment host."), ("RTX 3050 Laptop GPU", "RTX 3050 Laptop graphics processing unit (GPU)")],
37: [("REST API, a software request-and-reply interface", "REST application programming interface (API), a software request-and-reply interface")],
38: [("rated sample rate", "rated sampling rate")],
43: [("a Q4_K_M package", "a package using the Q4_K_M quantization format")],
46: [("We have not shown that it wins a manufacturing-agent comparison.", "The upgrade has not yet been evaluated on manufacturing-agent tasks."), ("to the NPU", "to the neural processing unit (NPU)")],
49: [("It checks the statement and uses a table library to get SQLite rows.", "It checks SQL syntax by parsing the statement, then uses a table library to execute the query and return SQLite rows. This syntax check is not an authorization policy.")],
53: [("setting temperature and", "setting temperature targets and")],
56: [("A fluent answer", "A plausible answer")],
59: [("CPU time and running time", "CPU usage and execution time")],
62: [("small multilayer perceptron tested", "small multilayer perceptron (MLP), a neural network with connected layers, tested"), ("decision cutoff", "decision threshold")],
63: [("Small memory buffers", "In-memory buffers")],
64: [("Crow-foot line ends show", "Crow's foot notation shows")],
68: [("Three positive visual frames or five defective acoustic events", "Three consecutive positive visual frames or five consecutive defective acoustic events")],
73: [("The waves and G-code", "The recorded waveforms and G-code")],
75: [("Missing or old records remain marked as such. The system does not replace them with unrelated data.", "The association design must retain missing and stale-record status rather than replace absent observations with unrelated data.")],
79: [("Selecting a new model does not turn an older image or deployment log into a new-model result.", "Recognition upgrades are evaluated separately from the retained interface examples.")],
84: [("The original dataset preparation uses", "Recorded dataset-preparation settings include"), ("It also changes saturation", "They also include saturation changes"), ("brightness by ±24%, with blur", "brightness changes of ±24%, with blur")],
93: [("We therefore do not claim that the camera and AE benchmarks cover the same print jobs. We also do not give them equal print counts.", "The collections do not provide a verified paired set of print jobs.")],
96: [("We compare YOLO11n and YOLO26n detectors", "We compare pretrained YOLO11n and YOLO26n detectors"), ("Early stopping has patience eight.", "Early stopping allows eight epochs without improvement."), ("Classifiers use 224-pixel inputs, and detectors use 640-pixel inputs.", "Classifiers use 224 × 224-pixel inputs, and detectors use 640 × 640-pixel inputs.")],
102: [("keeps candidate detections with confidence at least 0.001", "uses a confidence cutoff of 0.001 for candidate detections")],
103: [("Intersection over union (IoU) measures box overlap. AP50 uses IoU 0.50.", "Intersection over union (IoU) is the overlapping box area divided by the union of the box areas. AP50 is average precision at an IoU threshold of 0.50.")],
106: [("a small multilayer perceptron (MLP). An MLP is a neural network with connected layers.", "a small MLP."), ("Chen and Guestrin 2016", "T. Chen and Guestrin 2016")],
107: [("ONNX, a model file format", "Open Neural Network Exchange (ONNX), a model file format"), ("The deployed file", "The exported model package")],
111: [("Recall is the share of one true class that the model finds.", "Precision is the share of a class's predictions that are correct. Recall is the share of one true class that the model finds."), ("For F1,", "F1 combines precision and recall:")],
112: [("normal observations correctly rejected as defects", "normal observations correctly classified as normal"), ("AUROC summarizes", "Area under the receiver operating characteristic curve (AUROC) summarizes"), ("across cutoffs.", "across decision thresholds. The receiver operating characteristic (ROC) curve plots sensitivity against the false-positive rate.")],
117: [("correct record selection", "exact record-set correctness")],
123: [("A small file is therefore not always the fastest model.", "A smaller checkpoint does not necessarily mean faster inference.")],
130: [("refer to detected objects", "are object-level results, including missed labeled objects in recall")],
147: [("A host saving timestamp", "A host insertion timestamp, added when the record is stored,")],
154: [("Repeated-observation rules", "Consecutive-observation rules")],
158: [("should first fix the agent settings", "should first hold the agent settings"), ("models and decision rules.", "models and decision rules constant.")],
160: [("Returned measurements", "Returned observations")],
161: [("The AMD computer hosts the local language model.", "The AMD computer is assigned the local language-model deployment role.")],
}
CAPTIONS = {36:"Hardware setup",45:"Agent workflow",55:"Tool calls",66:"Data links",
            72:"Sensing and toolpath association",86:"Model workflows",128:"Visual results",140:"Acoustic results"}
LINKS = {"Consolidated evidence repository":URL,
         "MultiAgent SCADA":"https://github.com/bensidhom/MultiAgent_SCADA",
         "AI-driven analytics and digital twins":"https://github.com/bensidhom/Novel-Cyber-Infrastructure-Enabling-AI-Driven-Analytics-and-Digital-Twins-for-Real-Time-3D-Printing",
         "Acoustic emission monitoring at the edge":"https://github.com/bensidhom/Acoustic-Emission-Monitoring-at-the-Edge-for-Real-Time-Artificial-Intelligence-Applications"}

def visible(p):
    return "".join(p.xpath(".//w:t/text() | .//m:t/text()", namespaces=NS))

def tokenized(p):
    temp = deepcopy(p)
    for index, math in enumerate(temp.xpath("./m:oMath", namespaces=NS)):
        math.getparent().replace(math, run(f"[[MATH:{index}]]"))
    return visible(temp)

def run(text, props=None, italic=False):
    node = etree.Element(W+"r")
    if props is not None:
        node.append(deepcopy(props))
    if italic:
        rp = node.find(W+"rPr")
        if rp is None:
            rp = etree.SubElement(node,W+"rPr")
        etree.SubElement(rp,W+"i")
    t = etree.SubElement(node,W+"t")
    t.set(XMLSPACE,"preserve")
    t.text = text
    return node

def normalize_citations(text):
    text = text.replace("Chen et al. (2023)", "L. Chen et al. (2023)")
    # These suffixes follow title order: Hardware, Qwen3.6, Qwen3-Coder.
    text = text.replace("Ollama n.d.-a", "Ollama n.d.-TMP").replace("Ollama n.d.-c", "Ollama n.d.-a").replace("Ollama n.d.-TMP", "Ollama n.d.-c")
    pattern = re.compile(r"\(([^()]*(?:\b(?:19|20)\d{2}\b|n\.d\.)[^()]*)\)")
    def format_group(match):
        content = match.group(1)
        # Numeric method settings, not author-date citations, must remain untouched.
        if not re.match(r"(?:[A-ZÀ-Ž][\wÀ-ž .&'-]+)\s+(?:19|20|n\.d\.)", content):
            return match.group(0)
        entries = []
        for item in content.split("; "):
            item = item.replace(" and ", " & ")
            item = re.sub(r"\s+(?=(?:19|20)\d{2}|n\.d\.)", ", ", item, count=1)
            item = item.replace("Hugging Face, 2025, n.d.", "Hugging Face, n.d., 2025")
            entries.append(item)
        return "(" + "; ".join(sorted(entries, key=lambda s:re.sub(r"^[A-Z]\. ", "", s).casefold())) + ")"
    return pattern.sub(format_group, text)

def rewrite(p, text, relationships):
    math = p.xpath("./m:oMath", namespaces=NS)
    fields = p.xpath("./w:fldSimple", namespaces=NS)
    expected = [re.search(r"REF\s+(\S+)", f.get(W+"instr")).group(1) for f in fields]
    targets = [("fig" if k.startswith("Figure") else "tbl")+n for k,n in re.findall(r"\b(Figure|Figures|Table) (\d+)",text)]
    assert expected == targets, (expected, targets)
    assert len(re.findall(r"\[\[MATH:\d+\]\]",text)) == len(math)
    props = p.find("./w:r/w:rPr", namespaces=NS)
    special = p.xpath("./w:commentRangeStart | ./w:commentRangeEnd | ./w:r[w:commentReference]",namespaces=NS)
    for child in list(p):
        if child.tag != W+"pPr":
            p.remove(child)
    for child in special:
        if child.tag == W+"commentRangeStart":
            p.append(deepcopy(child))
    token = re.compile(r"\[\[MATH:(\d+)\]\]|\[\[LINK:([^\]]+)\]\]|\b(Figure|Figures|Table) (\d+)")
    cursor, field_index = 0, 0
    ids = {r.get("Target"):r.get("Id") for r in relationships}
    for match in token.finditer(text):
        if match.start()>cursor:
            p.append(run(text[cursor:match.start()],props))
        if match.group(1) is not None:
            p.append(deepcopy(math[int(match.group(1))]))
        elif match.group(2):
            label = match.group(2)
            url = LINKS[label]
            if url not in ids:
                rid = "rIdPublication" + str(len(relationships)+1)
                rel = etree.SubElement(relationships,"{http://schemas.openxmlformats.org/package/2006/relationships}Relationship")
                rel.set("Id",rid)
                rel.set("Type",NS["r"]+"/hyperlink")
                rel.set("Target",url)
                rel.set("TargetMode","External")
                ids[url] = rid
            link = etree.SubElement(p,W+"hyperlink")
            link.set(R+"id",ids[url])
            link.append(run(label,props))
        else:
            p.append(run(match.group(3)+" ",props))
            p.append(deepcopy(fields[field_index]))
            field_index += 1
        cursor = match.end()
    if cursor < len(text):
        p.append(run(text[cursor:],props))
    for child in special:
        if child.tag != W+"commentRangeStart":
            p.append(deepcopy(child))

def ref_text(text):
    if text.startswith("Ollama."):
        text = text.replace("(n.d.-a)","(n.d.-TMP)").replace("(n.d.-c)","(n.d.-a)").replace("(n.d.-TMP)","(n.d.-c)")
    if text.startswith("Ben Sidhom"):
        replacements = {"MultiAgent_SCADA":"b", "Novel Cyber":"c", "Acoustic Emission":"a"}
        for title, suffix in replacements.items():
            if title in text:
                text = re.sub(r"n\.d\.-[abc]", "n.d.-"+suffix,text)
    text = text.replace("MULTISENSOR FUSION-BASED DIGITAL TWIN IN ADDITIVE MANUFACTURING FOR IN-SITU QUALITY MONITORING AND DEFECT CORRECTION", "Multisensor fusion-based digital twin in additive manufacturing for in-situ quality monitoring and defect correction")
    text = text.replace("Automated Quality and Process Control for Additive Manufacturing using Deep Convolutional Neural Networks", "Automated quality and process control for additive manufacturing using deep convolutional neural networks")
    text = text.replace("Initial Work on the Characterization of Additive Manufacturing (3D Printing) Using Software Image Analysis", "Initial work on the characterization of additive manufacturing (3D printing) using software image analysis")
    text = text.replace("Digital Twin in Industry: State-of-the-Art", "Digital twin in industry: State-of-the-art")
    text = text.replace("Filament Breakage Monitoring in Fused Deposition Modeling Using Acoustic Emission Technique", "Filament breakage monitoring in fused deposition modeling using acoustic emission technique")
    return text

def format_reference(p, text, relationships):
    # Keep every URL as an active hyperlink and italicize the venue (or standalone web title).
    urls = re.findall(r"https?://\S+",text)
    year = re.search(r"\((?:\d{4}|n\.d\.(?:-[abc])?)\)\. ",text)
    assert year and urls
    title_start = year.end()
    title_end = text.find(". ",title_start)
    after_title = text[title_end+2:]
    venue = re.match(r"([^,]+), \d",after_title)
    italic_start, italic_end = title_start,title_end
    if venue:
        italic_start = title_end+2
        italic_end = italic_start+len(venue.group(1))
    elif after_title.startswith("The International Journal of Advanced Manufacturing Technology."):
        italic_start=title_end+2
        italic_end=italic_start+len("The International Journal of Advanced Manufacturing Technology")
    props = p.find("./w:r/w:rPr",namespaces=NS)
    for child in list(p):
        if child.tag!=W+"pPr":
            p.remove(child)
    url = urls[-1]
    link_start = text.rfind(url)
    cuts = sorted({0,italic_start,italic_end,link_start,len(text)})
    ids = {r.get("Target"):r.get("Id") for r in relationships}
    assert url in ids, url
    for start, end in zip(cuts,cuts[1:]):
        node = run(text[start:end],props, start>=italic_start and end<=italic_end)
        if start==link_start:
            link = etree.SubElement(p,W+"hyperlink")
            link.set(R+"id",ids[url])
            link.append(node)
        else:
            p.append(node)

def build(source, output):
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with ZipFile(source) as zipped:
        infos = zipped.infolist()
        parts = {i.filename:zipped.read(i.filename) for i in infos}
    doc = etree.fromstring(parts["word/document.xml"])
    baseline = deepcopy(doc)
    rels = etree.fromstring(parts["word/_rels/document.xml.rels"])
    ps = doc.xpath("/w:document/w:body/w:p",namespaces=NS)
    assert len(ps)==232 and visible(ps[167])=="References"
    changes=[]
    for index,p in enumerate(ps[:167]):
        if p.xpath(".//w:drawing",namespaces=NS):
            continue
        old = tokenized(p)
        text = FULL.get(index,old)
        for before, after in EDITS.get(index,[]):
            assert before in text, (index,before)
            text=text.replace(before,after)
        text=normalize_citations(text)
        if index in CAPTIONS or index==50:
            new_caption=CAPTIONS[index] if index in CAPTIONS else FULL[index]
            ends=p.xpath("./w:bookmarkEnd",namespaces=NS)
            assert len(ends)==1
            position=list(p).index(ends[0])+1
            for child in list(p)[position:]:
                p.remove(child)
            p.append(run(" "+new_caption))
        elif text!=old:
            rewrite(p,text,rels)
        if tokenized(p)!=old:
            changes.append({"paragraph":index,"before":old,"after":tokenized(p)})
    refs=[ref_text(visible(p)) for p in ps[168:]]
    # Same-author works: no date comes first, then date; suffixes order by title.
    for author in ("Ben Sidhom", "Ollama", "Hugging Face"):
        positions=[i for i,text in enumerate(refs) if text.startswith(author)]
        ordered=sorted((refs[i] for i in positions),key=lambda t:(0 if "(n.d." in t else 1,re.search(r"\(([^)]+)\)",t).group(1)))
        for index,text in zip(positions,ordered):
            refs[index]=text
    for p,text in zip(ps[168:],refs):
        format_reference(p,text,rels)
    for selector in ("//w:tbl","//w:drawing","//m:oMath","//w:fldSimple","//w:bookmarkStart","//w:bookmarkEnd","//w:sectPr","//w:commentReference"):
        assert [etree.tostring(n) for n in baseline.xpath(selector,namespaces=NS)]==[etree.tostring(n) for n in doc.xpath(selector,namespaces=NS)],selector
    assert not any("Ben Sidhom" in visible(p) and "n.d." in visible(p) for p in ps[:167])
    assert not any("INSERT PERMANENT" in visible(p) for p in ps)
    assert 150<=len(visible(ps[3]).split())<=250
    parts["word/document.xml"]=etree.tostring(doc,encoding="UTF-8",xml_declaration=True,standalone=True)
    parts["word/_rels/document.xml.rels"]=etree.tostring(rels,encoding="UTF-8",xml_declaration=True,standalone=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(output,"w") as zipped:
        for info in infos:
            zipped.writestr(info,parts[info.filename])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
    (ROOT/"paper/paragraph_revision_log.json").write_text(json.dumps(changes,ensure_ascii=False,indent=2),encoding="utf-8")
    (ROOT/"paper/manuscript_paragraphs.md").write_text("\n\n".join(f"<!-- paragraph {i} -->\n"+visible(p) for i,p in enumerate(ps) if visible(p)),encoding="utf-8")
    report={"source_preserved":True,"changed_paragraphs":len(changes),"references_formatted":64,"figures_preserved":8,"tables_preserved":8,"native_math_preserved":3,"abstract_words":len(visible(ps[3]).split()),"repository_url":URL,"visual_review":"pending"}
    (ROOT/"paper/revision_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=ROOT/"paper/source_before_revision.docx")
    parser.add_argument("--output",type=Path,default=ROOT/"paper/JIM_revised_manuscript_v3.docx")
    args=parser.parse_args()
    build(args.source,args.output)
