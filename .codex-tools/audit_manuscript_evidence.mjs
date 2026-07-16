import fs from 'node:fs';
import * as pdfjsLib from 'pdfjs-dist/legacy/build/pdf.mjs';

const pdfPath = process.argv[2];
const pdf = await pdfjsLib.getDocument({ data: new Uint8Array(fs.readFileSync(pdfPath)), disableWorker: true }).promise;
const attachments = await pdf.getAttachments();
const metadata = await pdf.getMetadata();

function erf(x) {
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const t = 1 / (1 + p * x);
  const y = 1 - (((((a5*t+a4)*t)+a3)*t+a2)*t+a1)*t*Math.exp(-x*x);
  return sign * y;
}
const phi = z => 0.5 * (1 + erf(z / Math.sqrt(2)));
function minWilcoxonP(n) {
  const exact = 2 / (2 ** n);
  const mean = n * (n + 1) / 4;
  const sd = Math.sqrt(n * (n + 1) * (2 * n + 1) / 24);
  const asymptotic = 2 * (1 - phi(mean / sd));
  return { n, exact, asymptotic };
}

const report = {
  pdfPages: pdf.numPages,
  attachments: attachments ? Object.keys(attachments) : [],
  metadata: metadata?.info ?? {},
  wilcoxonBounds: [10, 20, 30].map(minWilcoxonP),
  reportedPValues: {
    table5Minimum: 1.19e-6,
    table6Minimum: 2.55e-7,
  },
};
console.log(JSON.stringify(report, null, 2));
