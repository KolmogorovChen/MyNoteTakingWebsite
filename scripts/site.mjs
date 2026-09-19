import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

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
    return { text: title, link: url, draft: rel.startsWith('demo/'), size: Math.round(Buffer.byteLength(raw) / 1024) };
  });
  fs.cpSync(path.join(root, 'site', '.vitepress'), path.join(output, '.vitepress'), { recursive: true });
  fs.cpSync(path.join(root, 'site', 'public'), path.join(output, 'public'), { recursive: true });
  fs.writeFileSync(assetMapFile, JSON.stringify(assetMap, null, 2));
  fs.writeFileSync(path.join(output, '.vitepress', 'catalog.json'), JSON.stringify(entries, null, 2));
  const list = draft => entries.filter(e => e.draft === draft).map(e => `- [${e.text}](${encodeURI(e.link)}) <span class="note-size">${e.size} KB</span>`).join('\n');
  fs.writeFileSync(path.join(output, 'index.md'), `---\ntitle: 我的笔记\noutline: false\n---\n\n<div class="library-eyebrow">PERSONAL KNOWLEDGE LIBRARY</div>\n\n# 我的学习笔记\n\n<div class="library-intro">大模型理论与实践 · ${entries.length} 篇笔记</div>\n\n## 理论与学习路径\n\n${list(false)}\n\n## 草稿与补充\n\n${list(true)}\n`);
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
    if (!/\.(md|png|jpg|jpeg|svg|webp)$/i.test(filename)) return;
    clearTimeout(timer);
    timer = setTimeout(() => { try { generate(); } catch(e) { console.error(e); } }, 300);
  });
}
child.on('exit', code => { watcher?.close(); process.exit(code ?? 1); });
process.on('SIGINT', () => { watcher?.close(); child.kill(); });
