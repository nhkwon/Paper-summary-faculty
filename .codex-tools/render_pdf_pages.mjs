import fs from 'node:fs';
import path from 'node:path';
import * as pdfjsLib from 'pdfjs-dist/legacy/build/pdf.mjs';
import { createCanvas } from '@napi-rs/canvas';

const [input, outDir] = process.argv.slice(2);
fs.mkdirSync(outDir, { recursive: true });
const pdf = await pdfjsLib.getDocument({ data: new Uint8Array(fs.readFileSync(input)), disableWorker: true }).promise;
for (let i = 1; i <= pdf.numPages; i++) {
  const page = await pdf.getPage(i);
  const viewport = page.getViewport({ scale: 1.5 });
  const canvas = createCanvas(viewport.width, viewport.height);
  await page.render({ canvasContext: canvas.getContext('2d'), viewport }).promise;
  fs.writeFileSync(path.join(outDir, `page-${i}.png`), canvas.toBuffer('image/png'));
}
console.log(`Rendered ${pdf.numPages} pages`);
