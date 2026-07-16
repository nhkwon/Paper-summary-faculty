import fs from 'node:fs';
import PDFDocument from 'pdfkit';

const [input, output] = process.argv.slice(2);
const src = fs.readFileSync(input, 'utf8');
const isVerifiedAudit = /verified[_-]analysis/i.test(input);
const firstSectionIndex = src.indexOf('\\section*{');
const body = src.slice(firstSectionIndex, src.indexOf('\\end{document}'));
const title = clean((src.match(/\{\\Large\s+([^}]+)\}/) || [,'Mandatory Revision Guide'])[1]);
const subtitle = clean((src.match(/\{\\small\s+([^}]+)\}/) || [,'Blue text indicates suggested replacement or addition.'])[1]);

function clean(s) {
  return s
    .replace(/\\%/g, '%').replace(/\\&/g, '&').replace(/\\_/g, '_')
    .replace(/\\geq/g, '>=')
    .replace(/R\$\^2\$/g, 'R2').replace(/R\$\^\{2\}\$/g, 'R2')
    .replace(/\$([^$]+)\$/g, '$1')
    .replace(/\\textit\{([^{}]*)\}/g, '$1').replace(/\\textbf\{([^{}]*)\}/g, '$1')
    .replace(/``|''/g, '"').replace(/---/g, '-').replace(/--/g, '-')
    .replace(/\\item/g, '• ').replace(/\\begin\{itemize\}(\[[^\]]*\])?|\\end\{itemize\}/g, '')
    .replace(/\\[a-zA-Z]+(\[[^\]]*\])?/g, '').replace(/[{}]/g, '')
    .replace(/\\/g, '').replace(/\s+/g, ' ').trim();
}

const blocks = [];
const re = /\\section\*\{([^}]*)\}|\\(loc|current|replacewith|why)\{([\s\S]*?)\}(?=\n\\(?:loc|current|replacewith|why|section\*)|\n\n|$)/g;
let m;
while ((m = re.exec(body))) {
  if (m[1]) blocks.push({ type: 'section', text: clean(m[1]) });
  else blocks.push({ type: m[2], text: clean(m[3]) });
}

const doc = new PDFDocument({ size: 'LETTER', margins: { top: 46, bottom: 46, left: 50, right: 50 }, bufferPages: true, info: { Title: title } });
doc.pipe(fs.createWriteStream(output));
doc.font('Times-Bold').fontSize(17).text(title, { align: 'center' });
doc.moveDown(0.25).font('Times-Roman').fontSize(9.5).fillColor('#333333').text(subtitle, { align: 'center' });
doc.moveDown(1);

const sectionMatches = [...body.matchAll(/\\section\*\{([^}]*)\}/g)];
const introStart = sectionMatches[0]?.index + sectionMatches[0]?.[0].length;
const introEnd = sectionMatches[1]?.index ?? body.length;
const intro = clean(body.slice(introStart, introEnd));
let insertedIntro = false;
for (const b of blocks) {
  if (b.type === 'section') {
    doc.moveDown(0.8).fillColor('#111111').font('Times-Bold').fontSize(b.text.startsWith('Edit ') ? 13 : 14).text(b.text, { keepTogether: true });
    doc.moveDown(0.3);
    if (!insertedIntro) {
      doc.font('Times-Roman').fontSize(10.5).text(intro, { lineGap: 2 });
      insertedIntro = true;
    }
    if (b.text === 'Minor but Necessary Language Cleanup') {
      doc.font('Times-Roman').fontSize(10.5).text('Apply a final proofread for duplicated spaces, split words, and grammar, including: "the performance ... was evaluated" (page 7); "21 inputs and one output variable" and "cool-roof elements" (page 8); "developing" (page 11); "XGBoost supports classification, regression, ranking..." (page 11); and consistent use of "this study" rather than alternating with "this research."', { lineGap: 2 });
    }
    if (b.text === 'Do Not Finalize Until These Checks Pass') {
      const checks = [
        'Confirm training-only preprocessing, threshold computation, and hyperparameter selection from code or rerun the experiments.',
        'Add the component-cost ablation or narrow the "early-stage" claim.',
        'Recalculate Tables 5-6 with a clearly paired design and multiplicity control.',
        'Make all abstract, results, implications, and conclusion claims agree with Table 4.',
        'Complete the reference list and run a final visual proofread of equations, symbols, tables, and bullets.'
      ];
      for (const item of checks) doc.font('Times-Roman').fontSize(10.5).text(`• ${item}`, { indent: 12, lineGap: 2 });
    }
  } else {
    const label = isVerifiedAudit
      ? { loc: 'Location:', current: 'Verified evidence:', replacewith: 'Replace with / Required action:', why: 'Audit conclusion:' }[b.type]
      : { loc: 'Location:', current: 'Current context:', replacewith: 'Replace with / Add:', why: 'Why this edit:' }[b.type];
    doc.fillColor('#111111').font('Times-Bold').fontSize(10.5).text(label, { continued: b.type !== 'replacewith' });
    if (b.type === 'replacewith') doc.moveDown(0.1);
    doc.fillColor(b.type === 'replacewith' ? '#0645D6' : '#111111').font('Times-Roman').fontSize(10.5).text((b.type === 'replacewith' ? '' : ' ') + b.text, { lineGap: 2 });
    doc.moveDown(0.35);
  }
}

const range = doc.bufferedPageRange();
for (let i = range.start; i < range.start + range.count; i++) {
  doc.switchToPage(i);
  const oldBottomMargin = doc.page.margins.bottom;
  doc.page.margins.bottom = 0;
  doc.fillColor('#666666').font('Times-Roman').fontSize(8).text(String(i + 1), 0, 760, { width: 612, align: 'center', lineBreak: false });
  doc.page.margins.bottom = oldBottomMargin;
}
doc.end();
