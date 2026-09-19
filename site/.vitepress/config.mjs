import { defineConfig } from 'vitepress';
import fs from 'node:fs';
const catalog = JSON.parse(fs.readFileSync(new URL('./catalog.json', import.meta.url), 'utf8'));
const base = process.env.SITE_BASE || '/';
export default defineConfig({
  base,
  lang: 'zh-CN',
  title: '我的笔记',
  description: '个人大模型学习笔记',
  head: [['link', { rel: 'icon', href: `${base}favicon.svg` }]],
  lastUpdated: false,
  markdown: {
    math: true,
    config(md) {
      const fence = md.renderer.rules.fence;
      md.renderer.rules.fence = (tokens, idx, options, env, self) => tokens[idx].info.trim() === 'mermaid'
        ? `<MermaidDiagram source="${encodeURIComponent(tokens[idx].content)}" />`
        : fence(tokens, idx, options, env, self);
    }
  },
  themeConfig: {
    logo: '/favicon.svg',
    nav: [{ text: '全部笔记', link: '/' }],
    sidebar: [
      { text: '理论与学习路径', items: catalog.filter(e => !e.draft).map(({text,link}) => ({text,link})) },
      { text: '草稿与补充', collapsed: true, items: catalog.filter(e => e.draft).map(({text,link}) => ({text,link})) }
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
