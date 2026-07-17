from __future__ import annotations

import argparse
import datetime as dt
import os
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def x(text: str) -> str:
    return escape(text, {'"': "&quot;"})


def run(
    text: str,
    *,
    bold: bool = False,
    italic: bool = False,
    color: str | None = None,
    size_half_points: int | None = None,
    font: str | None = None,
) -> str:
    props: list[str] = []
    if font:
        props.append(
            f'<w:rFonts w:ascii="{x(font)}" w:hAnsi="{x(font)}" '
            'w:eastAsia="Malgun Gothic"/>'
        )
    if bold:
        props.append("<w:b/>")
    if italic:
        props.append("<w:i/>")
    if color:
        props.append(f'<w:color w:val="{color}"/>')
    if size_half_points:
        props.append(f'<w:sz w:val="{size_half_points}"/><w:szCs w:val="{size_half_points}"/>')
    rpr = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{x(text)}</w:t></w:r>'


def paragraph(
    content: str,
    *,
    style: str | None = None,
    keep_next: bool = False,
    alignment: str | None = None,
    before: int | None = None,
    after: int | None = None,
) -> str:
    props: list[str] = []
    if style:
        props.append(f'<w:pStyle w:val="{style}"/>')
    if keep_next:
        props.append("<w:keepNext/>")
    if alignment:
        props.append(f'<w:jc w:val="{alignment}"/>')
    if before is not None or after is not None:
        bits = []
        if before is not None:
            bits.append(f'w:before="{before}"')
        if after is not None:
            bits.append(f'w:after="{after}"')
        props.append(f"<w:spacing {' '.join(bits)}/>")
    ppr = f"<w:pPr>{''.join(props)}</w:pPr>" if props else ""
    return f"<w:p>{ppr}{content}</w:p>"


def cell(text: str, width: int, *, header: bool = False, file_text: bool = False) -> str:
    shade = '<w:shd w:val="clear" w:color="auto" w:fill="E8EEF5"/>' if header else ""
    tc_pr = (
        f'<w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{shade}'
        '<w:vAlign w:val="center"/></w:tcPr>'
    )
    if header:
        content = run(text, bold=True, color="1F3A5F", size_half_points=19)
        p = paragraph(content, style="TableHeader")
    elif file_text:
        content = run(text, bold=True, color="0B2545", size_half_points=18, font="Consolas")
        p = paragraph(content, style="TableText")
    else:
        content = run(text, size_half_points=19)
        p = paragraph(content, style="TableText")
    return f"<w:tc>{tc_pr}{p}</w:tc>"


def table(rows: list[tuple[str, str, str]]) -> str:
    widths = (1440, 4680, 3240)
    borders = "".join(
        f'<w:{edge} w:val="single" w:sz="4" w:space="0" w:color="B8C4D2"/>'
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV")
    )
    tbl_pr = (
        '<w:tblPr><w:tblW w:w="9360" w:type="dxa"/>'
        '<w:tblInd w:w="120" w:type="dxa"/>'
        '<w:tblLayout w:type="fixed"/>'
        f'<w:tblBorders>{borders}</w:tblBorders>'
        '<w:tblCellMar>'
        '<w:top w:w="80" w:type="dxa"/><w:left w:w="120" w:type="dxa"/>'
        '<w:bottom w:w="80" w:type="dxa"/><w:right w:w="120" w:type="dxa"/>'
        '</w:tblCellMar></w:tblPr>'
    )
    grid = '<w:tblGrid>' + ''.join(f'<w:gridCol w:w="{w}"/>' for w in widths) + '</w:tblGrid>'
    header = (
        '<w:tr><w:trPr><w:tblHeader/></w:trPr>'
        + cell("구분", widths[0], header=True)
        + cell("최신 파일명 또는 폴더명", widths[1], header=True)
        + cell("저장 위치", widths[2], header=True)
        + '</w:tr>'
    )
    body = []
    for category, filename, location in rows:
        body.append(
            '<w:tr>'
            + cell(category, widths[0])
            + cell(filename, widths[1], file_text=True)
            + cell(location, widths[2])
            + '</w:tr>'
        )
    return f"<w:tbl>{tbl_pr}{grid}{header}{''.join(body)}</w:tbl>"


def styles_xml() -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W_NS}">
  <w:docDefaults>
    <w:rPrDefault><w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Malgun Gothic"/>
      <w:sz w:val="22"/><w:szCs w:val="22"/><w:color w:val="222222"/>
      <w:lang w:val="en-US" w:eastAsia="ko-KR"/>
    </w:rPr></w:rPrDefault>
    <w:pPrDefault><w:pPr><w:spacing w:before="0" w:after="120" w:line="300" w:lineRule="auto"/></w:pPr></w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/><w:qFormat/>
    <w:pPr><w:spacing w:before="0" w:after="120" w:line="300" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Malgun Gothic"/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:next w:val="Subtitle"/><w:qFormat/>
    <w:pPr><w:keepNext/><w:spacing w:before="0" w:after="120"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Malgun Gothic"/><w:b/><w:color w:val="0B2545"/><w:sz w:val="48"/><w:szCs w:val="48"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Subtitle">
    <w:name w:val="Subtitle"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/>
    <w:pPr><w:keepNext/><w:spacing w:before="0" w:after="240" w:line="280" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Malgun Gothic"/><w:color w:val="555555"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:uiPriority w:val="9"/>
    <w:pPr><w:keepNext/><w:keepLines/><w:outlineLvl w:val="0"/><w:spacing w:before="360" w:after="200"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Malgun Gothic"/><w:b/><w:color w:val="2E74B5"/><w:sz w:val="32"/><w:szCs w:val="32"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:uiPriority w:val="9"/>
    <w:pPr><w:keepNext/><w:keepLines/><w:outlineLvl w:val="1"/><w:spacing w:before="280" w:after="140"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Malgun Gothic"/><w:b/><w:color w:val="2E74B5"/><w:sz w:val="26"/><w:szCs w:val="26"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading3">
    <w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:uiPriority w:val="9"/>
    <w:pPr><w:keepNext/><w:keepLines/><w:outlineLvl w:val="2"/><w:spacing w:before="200" w:after="100"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Malgun Gothic"/><w:b/><w:color w:val="1F4D78"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="PathBlock">
    <w:name w:val="Repository Path"/><w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:before="60" w:after="180" w:line="280" w:lineRule="auto"/><w:ind w:left="180" w:right="180"/>
      <w:shd w:val="clear" w:color="auto" w:fill="F4F6F9"/>
      <w:pBdr><w:top w:val="single" w:sz="4" w:color="D7DEE8"/><w:left w:val="single" w:sz="4" w:color="D7DEE8"/><w:bottom w:val="single" w:sz="4" w:color="D7DEE8"/><w:right w:val="single" w:sz="4" w:color="D7DEE8"/></w:pBdr>
    </w:pPr>
    <w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:eastAsia="Malgun Gothic"/><w:color w:val="334155"/><w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="TableHeader">
    <w:name w:val="Table Header"/><w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:before="0" w:after="0" w:line="260" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>
    <w:rPr><w:b/><w:color w:val="1F3A5F"/><w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="TableText">
    <w:name w:val="Table Text"/><w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:before="0" w:after="0" w:line="270" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="GuidanceLine">
    <w:name w:val="Guidance Line"/><w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:before="0" w:after="80" w:line="290" w:lineRule="auto"/><w:ind w:left="120"/></w:pPr>
    <w:rPr><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>
  </w:style>
</w:styles>'''


def document_xml() -> str:
    rows = [
        ("논문 원본", "(4) Manuscript_260716-01.pdf", "저장소 루트"),
        ("수정 지침", "verified_corrected_analysis_revision_guide_v4_blue.pdf", "output/pdf"),
        ("분석 요약", "manuscript_analysis_revision_summary_2026-07-17.docx", "output"),
        ("분석 노트북", "construction_cost_models_corrected.ipynb", "저장소 루트"),
        ("Python 그림", "portable_corrected_cost_figures.py", "analysis"),
        ("분석 결과", "corrected_analysis_20260717", "results"),
    ]
    title = paragraph(run("최신 논문·분석 파일 안내"), style="Title")
    subtitle = paragraph(
        run("Paper-summary-faculty 저장소 | 기준일 2026-07-18"),
        style="Subtitle",
    )
    repo_heading = paragraph(run("기준 저장소"), style="Heading2")
    repo_path = paragraph(
        run(r"C:\Users\nhkwo\Documents\GitHub\Paper-summary-faculty", font="Consolas"),
        style="PathBlock",
    )
    files_heading = paragraph(run("최신 주요 파일"), style="Heading2")
    file_table = table(rows)
    guide_heading = paragraph(run("사용 기준"), style="Heading2")
    guide_1 = paragraph(
        run("원본 보존  ", bold=True, color="1F3A5F")
        + run("(4) Manuscript_260716-01.pdf는 원본 자료이므로 덮어쓰거나 직접 수정하지 않습니다."),
        style="GuidanceLine",
    )
    guide_2 = paragraph(
        run("현재 기준  ", bold=True, color="1F3A5F")
        + run("논문 문구의 교체·삭제 판단에는 verified_corrected_analysis_revision_guide_v4_blue.pdf를 사용합니다."),
        style="GuidanceLine",
    )
    guide_3 = paragraph(
        run("결과 해석  ", bold=True, color="1F3A5F")
        + run("Figures 6–7은 seed 42 예시이므로 모델 우월성의 일반적 근거로 제시하지 않습니다."),
        style="GuidanceLine",
    )
    body = ''.join(
        [
            title,
            subtitle,
            repo_heading,
            repo_path,
            files_heading,
            file_table,
            guide_heading,
            guide_1,
            guide_2,
            guide_3,
        ]
    )
    sect = '''<w:sectPr>
      <w:footerReference w:type="default" r:id="rId4"/>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>
      <w:cols w:space="720"/><w:docGrid w:linePitch="360"/>
    </w:sectPr>'''
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{W_NS}" xmlns:r="{R_NS}"><w:body>{body}{sect}</w:body></w:document>'''


def footer_xml() -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:ftr xmlns:w="{W_NS}" xmlns:r="{R_NS}">
  <w:p>
    <w:pPr><w:pBdr><w:top w:val="single" w:sz="4" w:space="4" w:color="D7DEE8"/></w:pBdr><w:jc w:val="right"/><w:spacing w:before="80" w:after="0"/></w:pPr>
    {run("Paper-summary-faculty · 2026-07-18 · ", color="6B7280", size_half_points=17)}
    <w:r><w:rPr><w:color w:val="6B7280"/><w:sz w:val="17"/><w:szCs w:val="17"/></w:rPr><w:fldChar w:fldCharType="begin"/></w:r>
    <w:r><w:rPr><w:color w:val="6B7280"/><w:sz w:val="17"/><w:szCs w:val="17"/></w:rPr><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>
    <w:r><w:rPr><w:color w:val="6B7280"/><w:sz w:val="17"/><w:szCs w:val="17"/></w:rPr><w:fldChar w:fldCharType="end"/></w:r>
  </w:p>
</w:ftr>'''


def build(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    now = dt.datetime(2026, 7, 18, 0, 0, tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")
    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
  <Override PartName="/word/fontTable.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.fontTable+xml"/>
  <Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>'''
    package_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''
    document_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/fontTable" Target="fontTable.xml"/>
  <Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>
</Relationships>'''
    settings = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="{W_NS}"><w:zoom w:percent="100"/><w:updateFields w:val="true"/><w:defaultTabStop w:val="720"/><w:characterSpacingControl w:val="doNotCompress"/></w:settings>'''
    font_table = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:fonts xmlns:w="{W_NS}">
  <w:font w:name="Calibri"><w:family w:val="swiss"/><w:pitch w:val="variable"/></w:font>
  <w:font w:name="Malgun Gothic"><w:family w:val="swiss"/><w:pitch w:val="variable"/></w:font>
  <w:font w:name="Consolas"><w:family w:val="modern"/><w:pitch w:val="fixed"/></w:font>
</w:fonts>'''
    core = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>최신 논문·분석 파일 안내</dc:title><dc:subject>Paper-summary-faculty 파일 위치 요약</dc:subject>
  <dc:creator>OpenAI Codex</dc:creator><cp:lastModifiedBy>OpenAI Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>'''
    app = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Office Word</Application><DocSecurity>0</DocSecurity><ScaleCrop>false</ScaleCrop><Company></Company><LinksUpToDate>false</LinksUpToDate><SharedDoc>false</SharedDoc><HyperlinksChanged>false</HyperlinksChanged><AppVersion>16.0000</AppVersion>
</Properties>'''
    parts = {
        "[Content_Types].xml": content_types,
        "_rels/.rels": package_rels,
        "word/document.xml": document_xml(),
        "word/styles.xml": styles_xml(),
        "word/settings.xml": settings,
        "word/fontTable.xml": font_table,
        "word/footer1.xml": footer_xml(),
        "word/_rels/document.xml.rels": document_rels,
        "docProps/core.xml": core,
        "docProps/app.xml": app,
    }
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in parts.items():
            zf.writestr(name, data.encode("utf-8"))

    with zipfile.ZipFile(output) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"Corrupt ZIP member: {bad}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.output.resolve())
    print(os.fspath(args.output.resolve()))


if __name__ == "__main__":
    main()
