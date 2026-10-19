import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createMarkdownRenderer } from 'vitepress';
import cjkStrong from '../site/.vitepress/cjk-strong.mjs';
import headingSections from '../site/.vitepress/heading-sections.mjs';

const md = await createMarkdownRenderer(process.cwd(), {
  math: true
});
const OriginalState = md.inline.State;
const originalScanDelims = OriginalState.prototype.scanDelims;
md.use(cjkStrong).use(headingSections);
const render = source => md.render(source, { relativePath: 'notes/test.md' });
const renderPlain = source => {
  const CjkState = md.inline.State;
  md.inline.State = OriginalState;
  try { return render(source); }
  finally { md.inline.State = CjkState; }
};

test('Chinese labels and parentheses render strong without inserted spaces', () => {
  const cases = [
    ['**核心问题：**数据的物理布局怎样影响 I/O？', '<strong>核心问题：</strong>数据的物理布局怎样影响 I/O？'],
    ['**输入范围：**D4 读的是 D3', '<strong>输入范围：</strong>D4 读的是 D3'],
    ['按**页（Page）**组织', '按<strong>页（Page）</strong>组织'],
    ['这是**（说明）**正文', '这是<strong>（说明）</strong>正文'],
    ['这是**(说明)**正文', '这是<strong>(说明)</strong>正文'],
    ['这是**“术语”**正文', '这是<strong>“术语”</strong>正文'],
    ['**完整的一句话。**因此继续。', '<strong>完整的一句话。</strong>因此继续。'],
    ['𠀀**（说明）**𠀀', '𠀀<strong>（说明）</strong>𠀀']
  ];
  for (const [source, expected] of cases) assert.equal(render(source), `<p>${expected}</p>\n`, source);
});

test('headings, lists, quotes, tables, links and nested emphasis retain markup', () => {
  const source = [
    '# **主题：**内容', '', '- **列表：**内容', '', '> **引用：**内容', '',
    '| 内容 |', '| --- |', '| **标签：**内容 |', '',
    '[**链接：**内容](https://example.com)', '', '**标签：**正文，**第二项：**内容', '',
    '**标签：*斜体*（说明）**内容', '', '***提示：***正文'
  ].join('\n');
  const html = render(source);
  assert.equal((html.match(/<strong>/g) || []).length, 9);
  assert.ok(html.includes('<details class="note-section"'));
  assert.ok(html.includes('<strong>标签：<em>斜体</em>（说明）</strong>内容'));
  assert.ok(html.includes('<em><strong>提示：</strong></em>正文'));
  assert.ok(html.includes('<a href="https://example.com"'));
});

test('code, explicit escapes, math, whitespace and ordinary Markdown keep their behavior', () => {
  const unchanged = [
    '`**核心问题：**数据`',
    '```md\n**核心问题：**数据\n```',
    '    **核心问题：**数据',
    String.raw`\*\*核心问题：\*\*数据`,
    '** 标签：**正文', '**标签： **正文', '**标签：\n\n**正文',
    '*标签：*正文', '__标签：__正文', 'foo__bar__baz',
    '**English:**text', 'before**(English)**after',
    '**粗体**与 *斜体*，~~删除~~和 ![图](/image.png)',
    '$x^{**2}$', '$$\nx^{**2}\n$$'
  ];
  for (const source of unchanged) assert.equal(render(source), renderPlain(source), source);
});

test('enabling CJK strong does not modify the shared inline state prototype', () => {
  assert.equal(OriginalState.prototype.scanDelims, originalScanDelims);
  assert.equal(renderPlain('**标签：**正文'), '<p>**标签：**正文</p>\n');
  assert.equal(render('**标签：**正文'), '<p><strong>标签：</strong>正文</p>\n');
});

test('malformed emphasis keeps its original pairing without disturbing adjacent valid strong', () => {
  const malformed = '**(2) ** 每个簇的`cluster_id` 取该连通分量里**字典序最小**的 `doc_id`，纯粹为了"给这个簇一个稳定、可复现的名字"，**不等于最终保留者**。';
  assert.equal(render(malformed), renderPlain(malformed));
  const rollback = '**标签：**正文，' + malformed;
  assert.equal(render(rollback), renderPlain(rollback));
  const code = render('` ** ** ` **标签：**正文');
  assert.ok(code.includes('<strong>标签：</strong>正文'));
});
