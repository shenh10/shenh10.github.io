import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import { readdirSync, readFileSync, existsSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const DOCS = join(dirname(fileURLToPath(import.meta.url)), '..')

// Both blog sidebars are derived from the posts themselves, so adding a post
// is writing the post. Hand-maintained lists had already drifted — the Chinese
// one listed two of seven, and neither was in date order.
//
// Each post declares `short` (English sidebar label) and `titleZh` (Chinese),
// since full titles are too long for a sidebar and the Chinese ones are not
// translations of the English. A one-field frontmatter read is enough here; no
// need to pull in a parser.
function readPosts() {
  const dir = join(DOCS, 'blog')
  const field = (src: string, key: string) => {
    const m = src.match(new RegExp(`^${key}:\\s*"?(.+?)"?\\s*$`, 'm'))
    return m ? m[1] : ''
  }
  return readdirSync(dir)
    .filter((f) => f.endsWith('.md') && f !== 'index.md')
    .map((f) => {
      const slug = f.replace(/\.md$/, '')
      const head = readFileSync(join(dir, f), 'utf-8').split('---')[1] ?? ''
      return {
        slug,
        date: field(head, 'date'),
        title: field(head, 'title'),
        short: field(head, 'short'),
        titleZh: field(head, 'titleZh'),
        hasZh: existsSync(join(DOCS, 'zh', 'blog', f)),
      }
    })
    .filter((p) => p.date)
    .sort((a, b) => (a.date < b.date ? 1 : -1))
}

const posts = readPosts()

const blogSidebarEn = [
  {
    text: 'Posts',
    items: [
      { text: 'All posts', link: '/blog/' },
      ...posts.map((p) => ({ text: p.short || p.title, link: `/blog/${p.slug}` })),
    ],
  },
]

// a post links at its Chinese version when there is one, the English otherwise
const blogSidebarZh = [
  {
    text: '博客文章',
    items: [
      { text: '所有文章', link: '/zh/blog/' },
      ...posts.map((p) => ({
        text: p.titleZh || p.short || p.title,
        link: p.hasZh ? `/zh/blog/${p.slug}` : `/blog/${p.slug}`,
      })),
    ],
  },
]

// English is the root locale — nearly all the writing here is in English —
// with Chinese under /zh/. The project teardowns are long-form Chinese and are
// not translated; they keep their unprefixed /projects/claude-code/ URLs and
// both locales link straight at them.

const claudeCodeSidebar = [
  {
    text: 'Claude Code Codebook',
    items: [
      { text: '概览', link: '/projects/claude-code/' },
    ],
  },
  {
    text: '基础架构',
    items: [
      { text: '01 - 项目概述', link: '/projects/claude-code/01_foundation' },
      { text: '02 - 整体架构', link: '/projects/claude-code/02_architecture' },
      { text: '03 - 业务工作流', link: '/projects/claude-code/03_workflow' },
      { text: '04 - 核心数据结构与算法', link: '/projects/claude-code/04_core_mechanisms' },
    ],
  },
  {
    text: '模块深潜',
    items: [
      { text: '05 - 工具系统', link: '/projects/claude-code/05_module_tool_system' },
      { text: '05 - 权限系统', link: '/projects/claude-code/05_module_permission' },
      { text: '05 - Agent 子进程', link: '/projects/claude-code/05_module_agent' },
      { text: '05 - MCP 协议集成', link: '/projects/claude-code/05_module_mcp' },
      { text: '05 - Bridge 通信层', link: '/projects/claude-code/05_module_bridge' },
      { text: '05 - 上下文与内存管理', link: '/projects/claude-code/05_module_context' },
    ],
  },
  {
    text: '总结评估',
    items: [
      { text: '06 - 原生模块与性能优化', link: '/projects/claude-code/06_native_modules' },
      { text: '07 - 架构师定论', link: '/projects/claude-code/07_evaluation' },
    ],
  },
]

export default withMermaid(defineConfig({
  title: 'Han Shen',

  themeConfig: {
    siteTitle: false,

    socialLinks: [
      { icon: 'github', link: 'https://github.com/shenh10' },
    ],

    search: {
      provider: 'local',
    },
  },

  locales: {
    root: {
      label: 'English',
      lang: 'en-US',
      description: "Han Shen's blog and project notes",

      themeConfig: {
        nav: [
          { text: 'Home', link: '/' },
          { text: 'Blog', link: '/blog/' },
          {
            text: 'Teardowns',
            items: [
              { text: 'All teardowns', link: '/projects/' },
              { text: 'Claude Code (中文)', link: '/projects/claude-code/' },
            ],
          },
        ],

        sidebar: {
          '/blog/': blogSidebarEn,
          '/projects/claude-code/': claudeCodeSidebar,
        },

        outline: { level: [2, 3], label: 'On this page' },

        footer: {
          message: 'Powered by VitePress',
          copyright: '© 2026 Han Shen',
        },
      },
    },

    zh: {
      label: '简体中文',
      lang: 'zh-CN',
      link: '/zh/',
      description: '申晗的个人博客与项目文档',

      themeConfig: {
        nav: [
          { text: '首页', link: '/zh/' },
          { text: '博客', link: '/zh/blog/' },
          {
            text: '源码剖析',
            items: [
              { text: '所有剖析', link: '/zh/projects/' },
              { text: 'Claude Code 源码剖析', link: '/projects/claude-code/' },
            ],
          },
        ],

        sidebar: {
          '/zh/blog/': blogSidebarZh,
          '/projects/claude-code/': claudeCodeSidebar,
        },

        outline: { level: [2, 3], label: '本页目录' },

        docFooter: { prev: '上一篇', next: '下一篇' },
        darkModeSwitchLabel: '外观',
        returnToTopLabel: '返回顶部',
        langMenuLabel: '切换语言',

        footer: {
          message: 'Powered by VitePress',
          copyright: '© 2026 Han Shen',
        },
      },
    },
  },

  markdown: {
    math: true,
  },

  mermaid: {},

  vite: {
    optimizeDeps: {
      include: ['mermaid'],
    },
  },
}))
