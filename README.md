# 我的笔记网站

原有 Markdown 是内容源，网站不会修改它们。根目录和子目录中的 Markdown 会自动出现在目录中；`demo` 目录归为草稿与补充。README.md、AGENTS.md、隐藏目录和网站程序目录不收录。

## 启动

本机已装好依赖，可以双击 `start-notes.cmd`，然后打开 http://127.0.0.1:5173 。关闭终端会停止网站。

安装 Node.js（建议 22 或更新的 LTS 版本）和 Git，然后在此目录执行：

```sh
npm install
npm run dev
```

浏览器访问 http://127.0.0.1:5173 。保持终端运行，按 Ctrl+C 停止。只监听本机，不对局域网或公网开放。当前没有登录系统，也没有发布到互联网。

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
