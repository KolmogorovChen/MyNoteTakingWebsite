// Operate on parsed links so examples in code blocks and escaped Markdown stay literal.
const anchorPattern = /<!--[\s\S]*?-->|<(script|style|pre|code)\b[^>]*>[\s\S]*?<\/\1\s*>|<a\b(?:[^"'<>]|"[^"]*"|'[^']*')*>/gi;
const attributePattern = name => new RegExp(`(\\s+${name}\\s*=\\s*)(?:"([^"]*)"|'([^']*)'|([^\\s"'=<>\x60]+))`, 'i');

function attribute(tag, name) {
  const match = tag.match(attributePattern(name));
  return match ? match[2] ?? match[3] ?? match[4] : undefined;
}

function setAttribute(tag, name, value, escape) {
  const replacement = ` ${name}="${escape(value)}"`;
  return attributePattern(name).test(tag)
    ? tag.replace(attributePattern(name), () => replacement)
    : tag.replace(/>$/, `${replacement}>`);
}

const safeRel = rel => [...new Set(`${rel || ''} noopener noreferrer`.trim().split(/\s+/))].join(' ');

export function transformHtmlAnchors(html, transform, escape = value => value) {
  return html.replace(anchorPattern, tag => {
    if (!/^<a\b/i.test(tag)) return tag;
    const href = attribute(tag, 'href');
    if (href === undefined) return tag;
    const resolved = transform(href);
    if (resolved === undefined) return tag;
    let result = setAttribute(tag, 'href', resolved, escape);
    result = setAttribute(result, 'target', '_blank', escape);
    return setAttribute(result, 'rel', safeRel(attribute(tag, 'rel')), escape);
  });
}

export default function noteAttachments(md, { links = {}, base = '/' } = {}) {
  const resolve = (href, env) => {
    const page = String(env.relativePath || '').replaceAll('\\', '/');
    const url = links[page]?.[md.normalizeLink(md.utils.unescapeAll(href))];
    return url === undefined ? undefined : `${base.replace(/\/$/, '')}${url}`;
  };
  const linkOpen = md.renderer.rules.link_open;
  md.renderer.rules.link_open = (tokens, index, options, env, self) => {
    const token = tokens[index];
    const url = resolve(token.attrGet('href') || '', env);
    if (url !== undefined) {
      token.attrSet('href', url);
      // A native PDF navigation must not be intercepted as a VitePress page route.
      token.attrSet('target', '_blank');
      token.attrSet('rel', safeRel(token.attrGet('rel')));
    }
    return linkOpen ? linkOpen(tokens, index, options, env, self) : self.renderToken(tokens, index, options);
  };
  for (const type of ['html_inline', 'html_block']) {
    const render = md.renderer.rules[type];
    md.renderer.rules[type] = (tokens, index, options, env, self) => {
      tokens[index].content = transformHtmlAnchors(tokens[index].content, href => resolve(href, env), md.utils.escapeHtml);
      return render ? render(tokens, index, options, env, self) : tokens[index].content;
    };
  }
}
