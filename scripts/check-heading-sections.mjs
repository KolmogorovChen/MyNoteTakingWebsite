import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createRequire } from 'node:module';
import { createMarkdownRenderer } from 'vitepress';
import headingSections from '../site/.vitepress/heading-sections.mjs';

const { parse } = createRequire(import.meta.resolve('vue'))('@vue/compiler-dom');
const md = await createMarkdownRenderer(process.cwd(), {
  math: true,
  headers: true,
  config: md => md.use(headingSections)
});
const render = (source, relativePath = 'notes/test.md') => md.render(source, { relativePath });
const elements = node => (node.children || []).filter(child => child.type === 1);
const sections = node => elements(node).filter(child => child.tag === 'details');
const body = section => elements(section).find(child => child.tag === 'div');
const rank = section => elements(elements(section)[0])[0].tag;
const unwrap = html => html.replace(/<details class="note-section" data-level="[1-6]" open><summary class="note-summary">\n|<\/summary><div class="note-section-body">\n|<\/div><\/details>\n/g, '');

test('nested sections stop at equal or higher headings, including skipped levels', () => {
  const html = render('# A\n\nintro\n\n## B\n\nb\n\n#### C\n\nc\n\n## D\n\nd\n\n# E\n\ne');
  const roots = sections(parse(html));
  assert.deepEqual(roots.map(rank), ['h1', 'h1']);
  const children = sections(body(roots[0]));
  assert.deepEqual(children.map(rank), ['h2', 'h2']);
  assert.deepEqual(sections(body(children[0])).map(rank), ['h4']);
  assert.equal((html.match(/<details /g) || []).length, 5);
  assert.equal((html.match(/ open>/g) || []).length, 5);
});

test('all six levels, Setext headings, empty sections and duplicate titles', () => {
  const html = render('Title\n=====\n\n## Same\n## Same\n### Three\n#### Four\n##### Five\n###### Six\n\nlast');
  parse(html);
  for (let level = 1; level <= 6; level++) assert.ok(html.includes(`data-level="${level}"`));
  assert.ok(html.includes('id="same"'));
  assert.ok(html.includes('id="same-1"'));
});

test('math, code, tables, images and heading markup render identically', () => {
  const source = [
    '# Technical **note** with $x^2$', '', 'Introduction.', '',
    '## Formula', '', '$$', '\\begin{aligned} a &= b \\\\ c &= d \\end{aligned}', '$$', '',
    '```python', '# Not a heading', 'print("## Still code")', '```', '',
    '    # Indented code', '', '~~~text', '### Also code', '~~~', '',
    '| Key | Value |', '| --- | --- |', '| a | `# inline` |', '',
    '![Diagram](/note-assets/example.png)', '',
    '```mermaid', 'graph LR', 'A --> B', '```', '', '## End', '', 'Text.'
  ].join('\n');
  const html = render(source);
  assert.equal((html.match(/<details /g) || []).length, 3);
  assert.ok(html.includes('<mjx-container'));
  assert.ok(html.includes('<table'));
  assert.ok(html.includes('<img'));
  assert.ok(html.includes('language-python'));
  assert.equal(unwrap(html), render(source, 'topics/test.md'));
});

test('headings inside quotes, lists and custom containers remain in their containers', () => {
  const source = '# Page\n\n> ## Quote\n>\n> text\n\nOutside quote.\n\n- ### List heading\n\n  item content\n\n- next item\n\n::: tip\n## Tip heading\n\ntip text\n:::\n\n## Next\n\nend';
  const html = render(source);
  const root = sections(parse(html))[0];
  const content = elements(body(root));
  assert.equal(sections(content.find(n => n.tag === 'blockquote')).length, 1);
  assert.equal(sections(elements(content.find(n => n.tag === 'ul'))[0]).length, 1);
  assert.equal(sections(content.find(n => n.tag === 'div')).length, 1);
  assert.deepEqual(sections(body(root)).map(rank), ['h2']);
  assert.equal(unwrap(html), render(source, 'topics/test.md'));
});

test('home and topic pages are unchanged; a long note has one disclosure per heading', () => {
  assert.ok(!render('# Home', 'index.md').includes('note-section'));
  assert.ok(!render('# Topic', 'topics/test.md').includes('note-section'));
  const source = '# Long note\n' + Array.from({ length: 1000 }, (_, i) => `\n## Section ${i}\n\nText.\n\n### Child ${i}\n\nMore text.\n`).join('');
  const html = render(source);
  assert.equal((html.match(/<details /g) || []).length, 2001);
  assert.equal((html.match(/<\/details>/g) || []).length, 2001);
});
