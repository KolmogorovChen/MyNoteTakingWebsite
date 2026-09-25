export function groupTopics(entries) {
  const groups = new Map();
  for (const entry of entries) {
    const parts = entry.relativePath.split('/');
    const folder = parts.length > 1 ? parts[0] : '';
    if (!groups.has(folder)) groups.set(folder, {
      text: folder === '' ? '未分类' : folder === 'demo' ? '草稿与补充' : folder,
      folder,
      link: '/topics/' + (folder ? 'folder-' + encodeURIComponent(folder) : 'uncategorized'),
      entries: []
    });
    groups.get(folder).entries.push({ ...entry, section: parts.slice(1, -1).join(' / ') });
  }
  return [...groups.values()].sort((a, b) => a.folder === '' ? 1 : b.folder === '' ? -1 : a.text.localeCompare(b.text, 'zh-CN'));
}

export function topicItems(topic) {
  const items = [];
  const sections = new Map();
  for (const entry of topic.entries) {
    const item = { text: entry.text, link: entry.link };
    if (!entry.section) { items.push(item); continue; }
    if (!sections.has(entry.section)) {
      const group = { text: entry.section, collapsed: false, items: [] };
      sections.set(entry.section, group);
      items.push(group);
    }
    sections.get(entry.section).items.push(item);
  }
  return items;
}
