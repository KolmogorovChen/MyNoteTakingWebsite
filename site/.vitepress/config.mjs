import { defineConfig } from 'vitepress';
import fs from 'node:fs';
import { groupTopics, topicItems } from './topics.mjs';
const catalog = JSON.parse(fs.readFileSync(new URL('./catalog.json', import.meta.url), 'utf8'));
const topics = groupTopics(catalog);
const mainTopics = topics.filter(topic => topic.folder !== 'demo');
const drafts = topics.find(topic => topic.folder === 'demo');
const base = process.env.SITE_BASE || '/';
export default defineConfig({
  base,
  lang: 'zh-CN',
  title: '我的笔记',
  description: '按主题整理的个人学习笔记',
  head: [['link', { rel: 'icon', href: `${base}favicon.svg` }]],
  lastUpdated: false,
  markdown: {
    math: true,
    config(md) {
      const htmlInline = md.renderer.rules.html_inline;
      md.renderer.rules.html_inline = (tokens, idx, options, env, self) => /^<\/?(?:EOS|END|BOS|PAD|UNK|MASK)>$/i.test(tokens[idx].content)
        ? md.utils.escapeHtml(tokens[idx].content)
        : htmlInline(tokens, idx, options, env, self);
      const fence = md.renderer.rules.fence;
      md.renderer.rules.fence = (tokens, idx, options, env, self) => tokens[idx].info.trim() === 'mermaid'
        ? `<MermaidDiagram source="${encodeURIComponent(tokens[idx].content)}" />`
        : fence(tokens, idx, options, env, self);
    }
  },
  themeConfig: {
    logo: '/favicon.svg',
    nav: [{ text: '全部主题', link: '/' }, ...(mainTopics.length ? [{ text: '按主题浏览', items: mainTopics.map(({text,link}) => ({text,link})) }] : [])],
    sidebar: [
      ...mainTopics.map(topic => ({ text: topic.text, link: topic.link, collapsed: false, items: topicItems(topic) })),
      ...(drafts ? [{ items: [{ text: drafts.text, link: drafts.link }] }] : [])
    ],
    outline: { level: [2, 3], label: '本页目录' },
    docFooter: { prev: '上一篇', next: '下一篇' },
    sidebarMenuLabel: '笔记目录', returnToTopLabel: '返回顶部', darkModeSwitchLabel: '切换深色模式',
    search: { provider: 'local', options: {
      miniSearch: { options: { tokenize: text => Array.from(new Intl.Segmenter('zh-CN', { granularity: 'word' }).segment(text)).filter(s => s.isWordLike).map(s => s.segment.toLowerCase()) } },
      locales: { root: { translations: { button: { buttonText: '搜索笔记', buttonAriaLabel: '搜索笔记' }, modal: { displayDetails: '显示详情', resetButtonTitle: '清空', backButtonTitle: '返回', noResultsText: '没有找到相关笔记', footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' } } } } }
    } }
  }
});
