import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { test } from 'node:test';
import { createMarkdownRenderer } from 'vitepress';
import { collectNoteAttachments } from './note-attachments.mjs';
import noteAttachments from '../site/.vitepress/note-attachments.mjs';

const md = await createMarkdownRenderer(process.cwd(), { math: true });
const originalRules = { ...md.renderer.rules };
const page = 'notes/course/nested/note.md';

function fixture(t) {
  const testDirectory = path.resolve('.notes-site');
  fs.mkdirSync(testDirectory, { recursive: true });
  const root = fs.mkdtempSync(path.join(testDirectory, '.attachment-tests-'));
  assert.equal(path.dirname(root), testDirectory);
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const file = path.join(root, 'course', 'nested', 'note.md');
  const pdf = path.join(root, 'course', 'notes', '课件 (v2)#1.pdf');
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.mkdirSync(path.dirname(pdf), { recursive: true });
  const bytes = Buffer.from('%PDF-1.7\nfixture bytes\0\xff\n%%EOF', 'latin1');
  fs.writeFileSync(pdf, bytes);
  const warnings = [];
  const options = { file, root, publicDir: path.join(root, 'generated-public'), warn: value => warnings.push(value) };
  return { root, file, pdf, bytes, warnings, options };
}

async function render(source, links, base = '/', relativePath = page) {
  // VitePress caches its renderer; restore rules so each case has its own mapping.
  Object.assign(md.renderer.rules, originalRules);
  md.use(noteAttachments, { links: { [page]: links }, base });
  try {
    return md.render(source, { relativePath });
  } finally {
    Object.assign(md.renderer.rules, originalRules);
  }
}

test('copies referenced PDFs once, preserves bytes and page/query fragments, and leaves source files intact', t => {
  const f = fixture(t);
  const source = '[课件](<../notes/课件 (v2)%231.pdf?download=1#page=4>)\n\n[复习][slides]\n\n[slides]: <../notes/课件 (v2)%231.pdf#page=19>\n';
  fs.writeFileSync(f.file, source);
  const links = collectNoteAttachments(md, source, f.options);
  assert.equal(Object.keys(links).length, 2);
  const direct = links[md.normalizeLink('../notes/课件 (v2)%231.pdf?download=1#page=4')];
  assert.match(direct, /^\/note-assets\/[\da-f]{20}\.pdf\?download=1#page=4$/);
  assert.ok(Object.values(links).some(value => value.endsWith('#page=19')));
  const names = fs.readdirSync(path.join(f.options.publicDir, 'note-assets'));
  assert.equal(names.length, 1);
  assert.deepEqual(fs.readFileSync(path.join(f.options.publicDir, 'note-assets', names[0])), f.bytes);
  assert.deepEqual(fs.readFileSync(f.pdf), f.bytes);
  assert.equal(fs.readFileSync(f.file, 'utf8'), source);
  assert.deepEqual(f.warnings, []);
});

test('direct and reference links use a native PDF tab and respect a non-root site base', async t => {
  const f = fixture(t);
  const source = '[Direct](<../notes/课件 (v2)%231.pdf#page=4>)\n\n[Reference][slides]\n\n[slides]: <../notes/课件 (v2)%231.pdf#page=19>';
  const links = collectNoteAttachments(md, source, f.options);
  const html = await render(source, links, '/library/');
  assert.equal((html.match(/href="\/library\/note-assets\/[\da-f]{20}\.pdf#page=(?:4|19)"/g) || []).length, 2);
  assert.equal((html.match(/target="_blank"/g) || []).length, 2);
  assert.equal((html.match(/rel="noopener noreferrer"/g) || []).length, 2);
  assert.ok(!html.includes('.pdf.html'));
});

test('HTML anchors support encoded paths, spaces, query entities and existing attributes', async t => {
  const f = fixture(t);
  const source = `<a class="slides" href='../notes/课件 (v2)%231.pdf?x=1&amp;y=2#page=4' target="_self" rel="nofollow">Inline</a>\n\n<div>\n<a href="/course/notes/课件 (v2)%231.pdf#page=19">Block</a>\n</div>`;
  const links = collectNoteAttachments(md, source, f.options);
  const html = await render(source, links, '/library/');
  assert.equal(Object.keys(links).length, 2);
  assert.match(html, /href="\/library\/note-assets\/[\da-f]{20}\.pdf\?x=1&amp;y=2#page=4"/);
  assert.match(html, /class="slides"/);
  assert.match(html, /rel="nofollow noopener noreferrer"/);
  assert.equal((html.match(/target="_blank"/g) || []).length, 2);
  assert.ok(!html.includes('target="_self"'));
});

test('examples, external links, Markdown pages and unrelated pages are not attachment routes', async t => {
  const f = fixture(t);
  const source = [
    '```md', '[example](missing.pdf)', '<a href="also-missing.pdf">example</a>', '```', '',
    '`[inline](inline-missing.pdf)`', '',
    '\\[escaped\\](escaped-missing.pdf)', '',
    '[web](https://example.com/slides.pdf#page=4)', '',
    '[protocol relative](//example.com/slides.pdf)', '',
    '[page](../other.md)', '',
    '<!-- <a href="comment.pdf">Comment</a> -->', '',
    '<script>const example = \'<a href="script.pdf">Sample</a>\';</script>'
  ].join('\n');
  assert.deepEqual(collectNoteAttachments(md, source, f.options), {});
  assert.deepEqual(f.warnings, []);
  const html = await render(source, {});
  assert.match(html, /href="https:\/\/example.com\/slides.pdf#page=4"/);
  assert.match(html, /href="(?:\.\/)?\.\.\/other.html"/);
  const href = '../notes/课件 (v2)%231.pdf#page=4';
  const mapping = { [md.normalizeLink(href)]: '/note-assets/test.pdf#page=4' };
  const unrelated = await render(`[PDF](<${href}>)`, mapping, '/', 'topics/course.md');
  assert.ok(!unrelated.includes('/note-assets/'));
});

test('missing and out-of-repository PDFs are reported and never copied', t => {
  const f = fixture(t);
  const links = collectNoteAttachments(md, '[Missing](missing.pdf)\n\n[Outside](../../../private.pdf)', f.options);
  assert.deepEqual(links, {});
  assert.equal(f.warnings.length, 2);
  assert.ok(f.warnings.some(value => value.includes('PDF 文件不存在')));
  assert.ok(f.warnings.some(value => value.includes('PDF 链接超出笔记目录')));
  assert.ok(!fs.existsSync(f.options.publicDir));
});
