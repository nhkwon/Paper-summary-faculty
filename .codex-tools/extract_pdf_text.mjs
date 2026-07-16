import fs from 'node:fs';
import * as pdfjsLib from 'pdfjs-dist/legacy/build/pdf.mjs';

const [input, output] = process.argv.slice(2);
if (!input || !output) throw new Error('Usage: node extract_pdf_text.mjs input.pdf output.txt');
const data = new Uint8Array(fs.readFileSync(input));
const pdf = await pdfjsLib.getDocument({ data, disableWorker: true }).promise;
const pages = [];
for (let i = 1; i <= pdf.numPages; i++) {
  const page = await pdf.getPage(i);
  const content = await page.getTextContent();
  let lastY = null;
  let text = '';
  for (const item of content.items) {
    const y = item.transform?.[5] ?? 0;
    if (lastY !== null && Math.abs(y - lastY) > 2) text += '\n';
    else if (text && !text.endsWith('\n') && !text.endsWith(' ')) text += ' ';
    text += item.str;
    lastY = y;
  }
  pages.push(`\n===== PAGE ${i} =====\n${text}\n`);
}
fs.writeFileSync(output, pages.join(''), 'utf8');
console.log(`Extracted ${pdf.numPages} pages to ${output}`);
