import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { groupTopics, topicItems } from '../site/.vitepress/topics.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.join(root, '.notes-site');
const assets = path.join(root, 'site', 'public', 'note-assets');
const skip = new Set(['node_modules', '.git', '.notes-site', '.local-runtime', '.sites-runtime', 'site', 'scripts']);
const warnings = new Set();
const assetMapFile = path.join(root, 'site', 'asset-map.json');
const assetMap = fs.existsSync(assetMapFile) ? JSON.parse(fs.readFileSync(assetMapFile, 'utf8')) : {};
function scan(dir = root) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap(e => {
    if (skip.has(e.name) || e.name.startsWith('.')) return [];
    const p = path.join(dir, e.name);
    return e.isDirectory() ? scan(p) : e.name.endsWith('.md') && e.name !== 'README.md' && e.name !== 'AGENTS.md' ? [p] : [];
  });
}
function asset(src, file) {
  if (/^(https?:|data:|\/\/)/i.test(src)) return src;
  let decoded;
  try { decoded = decodeURI(src); } catch { decoded = src; }
  const abs = path.isAbsolute(decoded) ? decoded : path.resolve(path.dirname(file), decoded);
  const key = crypto.createHash('sha256').update(src).digest('hex');
  if (!fs.existsSync(abs)) {
    if (assetMap[key] && fs.existsSync(path.join(assets, assetMap[key]))) return `/note-assets/${assetMap[key]}`;
    warnings.add(`图片不存在：${abs}`); return src;
  }
  const bytes = fs.readFileSync(abs);
  const name = crypto.createHash('sha256').update(bytes).digest('hex').slice(0, 20) + path.extname(abs);
  fs.mkdirSync(assets, { recursive: true });
  if (!fs.existsSync(path.join(assets, name))) fs.writeFileSync(path.join(assets, name), bytes);
  assetMap[key] = name;
  return `/note-assets/${name}`;
}
function generate() {
  warnings.clear();
  fs.mkdirSync(output, { recursive: true });
  const files = scan().sort((a,b) => a.localeCompare(b, 'zh-CN'));
  const previousCatalog = path.join(output, '.vitepress', 'catalog.json');
  if (fs.existsSync(previousCatalog)) {
    for (const old of JSON.parse(fs.readFileSync(previousCatalog, 'utf8'))) {
      const oldPage = path.resolve(output, '.' + old.link + '.md');
      const source = path.resolve(root, old.link.replace(/^\/notes\//, '') + '.md');
      if (oldPage.startsWith(output + path.sep) && !files.includes(source) && fs.existsSync(oldPage)) fs.unlinkSync(oldPage);
    }
  }
  const entries = files.map(file => {
    const rel = path.relative(root, file).replaceAll('\\', '/');
    const raw = fs.readFileSync(file, 'utf8');
    const title = raw.match(/^#\s+(.+)$/m)?.[1]?.trim() || path.basename(file, '.md');
    const url = '/notes/' + rel.replace(/\.md$/, '');
    let content = raw.replace(/<img\b[^>]*\bsrc=["']([^"']+)["'][^>]*>/gi, (_, src) => `![笔记插图](${asset(src, file)})`);
    content = content.replace(/!\[([^\]]*)\]\(([^\s)]+)(?:\s+"[^"]*")?\)/g, (_, alt, src) => `![${alt}](${src.startsWith('/note-assets/') ? src : asset(src, file)})`);
    const target = path.join(output, 'notes', rel);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, `---\ntitle: ${JSON.stringify(title)}\n---\n\n` + (/^#\s/m.test(content) ? '' : `# ${title}\n\n`) + content);
    return { text: title, link: url, relativePath: rel, size: Math.round(Buffer.byteLength(raw) / 1024) };
  });
  fs.cpSync(path.join(root, 'site', '.vitepress'), path.join(output, '.vitepress'), { recursive: true });
  fs.cpSync(path.join(root, 'site', 'public'), path.join(output, 'public'), { recursive: true });
  fs.writeFileSync(assetMapFile, JSON.stringify(assetMap, null, 2));
  fs.writeFileSync(path.join(output, '.vitepress', 'catalog.json'), JSON.stringify(entries, null, 2));
  const topics = groupTopics(entries);
  const mainTopics = topics.filter(topic => topic.folder !== 'demo');
  const drafts = topics.find(topic => topic.folder === 'demo');
  const topicDir = path.join(output, 'topics');
  fs.mkdirSync(topicDir, { recursive: true });
  const currentPages = new Set(topics.map(t => decodeURIComponent(t.link.split('/').pop()) + '.md'));
  for (const name of fs.readdirSync(topicDir)) {
    if (name.endsWith('.md') && !currentPages.has(name)) fs.unlinkSync(path.join(topicDir, name));
  }
  const label = text => text.replace(/[\\[\]<>]/g, c => ({'\\':'&#92;','[':'&#91;',']':'&#93;','<':'&lt;','>':'&gt;'}[c]));
  for (const topic of topics) {
    const list = topicItems(topic).map(item => item.items
      ? `\n## ${label(item.text)}\n\n` + item.items.map(e => `- [${label(e.text)}](${encodeURI(e.link)})`).join('\n')
      : `- [${label(item.text)}](${encodeURI(item.link)})`).join('\n');
    fs.writeFileSync(path.join(topicDir, decodeURIComponent(topic.link.split('/').pop()) + '.md'), `---\ntitle: ${JSON.stringify(topic.text)}\n---\n\n# ${label(topic.text)}\n\n${topic.entries.length} 篇笔记 · [全部主题](/)\n\n${list}\n`);
  }
  const overview = mainTopics.map(t => `## [${label(t.text)}](${t.link})\n\n${t.entries.length} 篇笔记\n\n` + t.entries.slice(0, 4).map(e => `- [${label(e.text)}](${encodeURI(e.link)})`).join('\n') + (t.entries.length > 4 ? `\n\n[查看全部笔记 →](${t.link})` : '')).join('\n\n');
  const draftLink = drafts ? `\n\n<div class="library-secondary">\n\n[草稿与补充](${drafts.link}) · ${drafts.entries.length} 篇\n\n</div>\n` : '';
  const mainCount = mainTopics.reduce((sum, topic) => sum + topic.entries.length, 0);
  fs.writeFileSync(path.join(output, 'index.md'), `---\ntitle: 我的笔记\noutline: false\n---\n\n<div class="library-eyebrow">PERSONAL KNOWLEDGE LIBRARY</div>\n\n# 我的学习笔记\n\n<div class="library-intro">${mainTopics.length} 个主题 · ${mainCount} 篇笔记</div>\n\n${overview || '还没有主题笔记。在项目目录中创建主题文件夹，并放入 Markdown 文件即可。'}${draftLink}\n`);
  for (const warning of warnings) console.warn(warning);
  console.log(`已生成 ${entries.length} 篇笔记。`);
}
generate();
const mode = process.argv[2] || 'dev';
const cli = path.join(root, 'node_modules', 'vitepress', 'bin', 'vitepress.js');
const args = [cli, mode, output, ...(mode === 'dev' ? ['--host', '127.0.0.1', '--port', '5173', '--strictPort'] : [])];
const child = spawn(process.execPath, args, { cwd: root, stdio: 'inherit' });
let watcher;
if (mode === 'dev') {
  let timer;
  watcher = fs.watch(root, { recursive: true }, (_, filename) => {
    if (!filename || filename.split(/[\\/]/).some(p => skip.has(p) || p.startsWith('.'))) return;
    // Directory moves must refresh the catalog as well as individual file edits.
    clearTimeout(timer);
    timer = setTimeout(() => { try { generate(); } catch(e) { console.error(e); } }, 300);
  });
}
child.on('exit', code => { watcher?.close(); process.exit(code ?? 1); });
process.on('SIGINT', () => { watcher?.close(); child.kill(); });
