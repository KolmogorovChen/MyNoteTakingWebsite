# 我的笔记网站

原有 Markdown 是内容源，网站不会修改它们。一级文件夹自动成为主题，子文件夹成为主题内分类，每个主题都有独立目录页。根目录的笔记放在“未分类”，`demo` 仍显示为“草稿与补充”。没有 Markdown 的空文件夹不显示。README.md、AGENTS.md、隐藏目录和网站程序目录不收录。

## 按主题整理

在 Windows 文件资源管理器打开 `E:\AI_Learning\大模型理论学习`，右键空白处 → 新建文件夹，用主题名称命名，例如“大模型学习”“Python”“数学”。将对应的 `.md` 文件剪切到主题文件夹即可，也可以在主题内继续建子文件夹。不要把笔记放进 site、scripts、node_modules 或以点开头的目录。

```text
大模型理论学习/
├─ 大模型学习/
│  ├─ Transformer.md
│  └─ 预训练/
│     └─ 大语言模型预训练.md
├─ Python/
│  └─ 基础语法.md
├─ 数学/
│  └─ 概率论.md
├─ figures/
├─ site/                  网站程序，保留
├─ scripts/               构建程序，保留
└─ start-notes.cmd
```

上面是整理方式示例，不代表所有示例笔记已创建。已准备好空的“大模型学习”文件夹，现有笔记尚未移动。

移动笔记时注意：

- 图片相对路径是从 Markdown 所在位置计算的。若从根目录移到“大模型学习”，原来的 `figures/example.png` 应改为 `../figures/example.png`；移入两层文件夹则用 `../../figures/example.png`。也可以把 Markdown 和其相邻图片文件夹一起移动，保持相对位置。
- 文章互链同样需要按新位置修改，例如 `[Transformer](../Transformer.md)`。不要依赖本机绝对路径给新图片做引用。
- 文件夹或文件名变化会改变文章网址，旧收藏链接需要更新。主题目录、首页及搜索索引会随构建自动更新。
- 不用修改网站配置。双击 start-notes.cmd 检查本地效果；如果已有旧版服务运行，升级这次程序后先关闭再重新启动。
- 检查后在项目根目录执行下面的 Git 命令发布。仓库和网站公开，新放入主题文件夹的笔记也会公开。

## 启动

本机已装好依赖，可以双击 `start-notes.cmd`，然后打开 http://127.0.0.1:5173 。关闭终端会停止网站。

安装 Node.js（建议 22 或更新的 LTS 版本）和 Git，然后在此目录执行：

```sh
npm install
npm run dev
```

浏览器访问 http://127.0.0.1:5173 。保持终端运行，按 Ctrl+C 停止。本地预览只监听本机；线上网站通过 Cloudflare Pages 发布。

开发环境若没有 npm，可使用 `pnpm install` 和 `pnpm dev`。仓库提供 pnpm-lock.yaml，推荐 `pnpm install --frozen-lockfile` 复现依赖。

## 更新笔记

直接修改原来的 Markdown，或新增 Markdown 文件。运行中的网站会自动重新生成。图片推荐放入 figures 文件夹并使用相对路径，例如 `![示意图](figures/example.png)`。

生成时会将本地引用的图片复制到 `site/public/note-assets`（应提交到 Git），并在 `site/asset-map.json` 保留映射，以便换电脑后仍可使用已收录图片。原始笔记保持不变。新图片建议始终使用仓库内相对路径。

## Git

提交原始笔记、figures、site、scripts、package.json、pnpm-lock.yaml、pnpm-workspace.yaml、start-notes.cmd、.gitignore 和本说明。node_modules 和 .notes-site 是依赖或生成结果，不提交。

仓库：https://github.com/KolmogorovChen/MyNoteTakingWebsite

网站：https://my-note-taking-website.pages.dev/

仓库和网站均公开。推送 main 后，Cloudflare Pages 自动构建并发布全部收录的笔记及图片；本地运行仍使用 http://127.0.0.1:5173 。

```sh
git add .
git commit -m "更新笔记"
git push
```

## 构建

```sh
npm run build
npm run preview
```

静态网页位于 `.notes-site/.vitepress/dist`。支持中文全文搜索、MathJax 数学公式、Mermaid 流程图、深色模式和手机布局。
