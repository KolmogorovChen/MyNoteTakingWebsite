import fs from 'node:fs';
import path from 'node:path';
const dir = '.notes-site/.vitepress/dist';
function files(p) { return fs.readdirSync(p, { withFileTypes:true }).flatMap(e => e.isDirectory() ? files(path.join(p,e.name)) : [path.join(p,e.name)]); }
const pages = files(dir).filter(p => p.endsWith('.html'));
let formulas = 0, images = 0, pdfLinks = 0;
const missing = [];
for (const p of pages) {
  const html = fs.readFileSync(p,'utf8');
  formulas += (html.match(/class="MathJax"/g) || []).length;
  for (const m of html.matchAll(/src="(\/note-assets\/[^"#?]+)"/g)) {
    images++;
    if (!fs.existsSync(path.join(dir, m[1]))) missing.push(m[1]);
  }
  for (const m of html.matchAll(/href="([^"?#]*\/note-assets\/[^"?#]+\.pdf)(?:[?#][^"]*)?"/g)) {
    pdfLinks++;
    const pathname = decodeURIComponent(m[1]).replace(process.env.SITE_BASE || '/', '/');
    const file = path.join(dir, pathname);
    if (!fs.existsSync(file) || fs.readFileSync(file).subarray(0, 5).toString() !== '%PDF-') missing.push(m[1]);
  }
}
console.log(JSON.stringify({pages:pages.length,formulas,images,pdfLinks,missing},null,2));
if (missing.length || !formulas || !images) process.exitCode = 1;
