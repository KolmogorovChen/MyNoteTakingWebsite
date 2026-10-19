import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { transformHtmlAnchors } from '../site/.vitepress/note-attachments.mjs';

export function collectNoteAttachments(md, source, { file, root, publicDir, warn = console.warn }) {
  const links = {};
  const visit = href => {
    href = md.normalizeLink(md.utils.unescapeAll(href));
    if (/^(?:[a-z][a-z\d+.-]*:|\/\/|#)/i.test(href)) return;
    const [, pathname, suffix = ''] = href.match(/^([^?#]*)([?#].*)?$/);
    if (!/\.pdf$/i.test(pathname)) return;
    let decoded;
    try { decoded = decodeURIComponent(pathname); } catch { return; }
    const absolute = pathname.startsWith('/')
      ? path.resolve(root, `.${decoded}`)
      : path.resolve(path.dirname(file), decoded);
    const relative = path.relative(root, absolute);
    if (relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
      warn(`PDF 链接超出笔记目录：${file} -> ${href}`);
      return;
    }
    if (!fs.existsSync(absolute) || !fs.statSync(absolute).isFile()) {
      warn(`PDF 文件不存在：${file} -> ${href}`);
      return;
    }
    const bytes = fs.readFileSync(absolute);
    const name = `${crypto.createHash('sha256').update(bytes).digest('hex').slice(0, 20)}.pdf`;
    const destination = path.join(publicDir, 'note-assets', name);
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    if (!fs.existsSync(destination)) fs.writeFileSync(destination, bytes);
    links[href] = `/note-assets/${name}${suffix}`;
  };
  const walk = tokens => {
    for (const token of tokens) {
      if (token.type === 'link_open') visit(token.attrGet('href') || '');
      if (token.type === 'html_inline' || token.type === 'html_block') transformHtmlAnchors(token.content, visit);
      if (token.children) walk(token.children);
    }
  };
  walk(md.parse(source, {}));
  return links;
}
