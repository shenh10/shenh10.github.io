import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(defineConfig({
  title: 'Shen Han',
  description: 'Shen Han 的个人博客与项目文档',
  lang: 'zh-CN',

  themeConfig: {
    siteTitle: false,

    nav: [
      { text: '首页', link: '/' },
      { text: '博客', link: '/blog/' },
      {
        text: '项目',
        items: [
          { text: '所有项目', link: '/projects/' },
          { text: 'Claude Code 源码剖析', link: '/projects/claude-code/' },
        ],
      },
      { text: 'PaperCache', link: 'https://www.papercache.org/' },
      { text: '关于我', link: '/about' },
    ],

    sidebar: {
      '/blog/': [
        {
          text: '博客文章',
          items: [
            { text: '所有文章', link: '/blog/' },
          ],
        },
      ],

      '/projects/claude-code/': [
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
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/shenh10' },
    ],

    outline: {
      level: [2, 3],
      label: '本页目录',
    },

    search: {
      provider: 'local',
    },

    footer: {
      message: 'Powered by VitePress',
      copyright: '© 2026 Shen Han',
    },
  },

  mermaid: {},

  vite: {
    optimizeDeps: {
      include: ['mermaid'],
    },
  },
}))
