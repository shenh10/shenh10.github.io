import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

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
  title: 'Shen Han',

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
      description: "Shen Han's blog and project notes",

      themeConfig: {
        nav: [
          { text: 'Home', link: '/' },
          { text: 'Blog', link: '/blog/' },
          {
            text: 'Projects',
            items: [
              { text: 'All projects', link: '/projects/' },
              { text: 'Claude Code teardown (中文)', link: '/projects/claude-code/' },
            ],
          },
          { text: 'PaperCache', link: 'https://www.papercache.org/' },
          { text: 'About', link: '/about' },
        ],

        sidebar: {
          '/blog/': [
            {
              text: 'Posts',
              items: [
                { text: 'All posts', link: '/blog/' },
                { text: 'HuggingArch: Automating Model Architecture Analysis', link: '/blog/huggingarch' },
                { text: 'GPU-to-GPU Copy over PCIe', link: '/blog/gpu-d2d-pcie' },
                { text: 'GPU Clock Throttling', link: '/blog/gpu-throttling' },
                { text: 'DeepSeek Inference Efficiency (1): Throughput Ceiling', link: '/blog/ds-inference-1-throughput-ceiling' },
                { text: 'DeepSeek Inference Efficiency (2): Reverse-Engineering', link: '/blog/ds-inference-2-reverse-engineering' },
                { text: 'DeepSeek Inference Efficiency (3): Decode Generalization', link: '/blog/ds-inference-3-decode-generalization' },
              ],
            },
          ],
          '/projects/claude-code/': claudeCodeSidebar,
        },

        outline: { level: [2, 3], label: 'On this page' },

        footer: {
          message: 'Powered by VitePress',
          copyright: '© 2026 Shen Han',
        },
      },
    },

    zh: {
      label: '简体中文',
      lang: 'zh-CN',
      link: '/zh/',
      description: 'Shen Han 的个人博客与项目文档',

      themeConfig: {
        nav: [
          { text: '首页', link: '/zh/' },
          { text: '博客', link: '/zh/blog/' },
          {
            text: '项目',
            items: [
              { text: '所有项目', link: '/zh/projects/' },
              { text: 'Claude Code 源码剖析', link: '/projects/claude-code/' },
            ],
          },
          { text: 'PaperCache', link: 'https://www.papercache.org/' },
          { text: '关于我', link: '/zh/about' },
        ],

        sidebar: {
          '/zh/blog/': [
            {
              text: '博客文章',
              items: [
                { text: '所有文章', link: '/zh/blog/' },
                { text: 'HuggingArch：让模型 arch 分析自动化', link: '/zh/blog/huggingarch' },
              ],
            },
          ],
          '/projects/claude-code/': claudeCodeSidebar,
        },

        outline: { level: [2, 3], label: '本页目录' },

        docFooter: { prev: '上一篇', next: '下一篇' },
        darkModeSwitchLabel: '外观',
        returnToTopLabel: '返回顶部',
        langMenuLabel: '切换语言',

        footer: {
          message: 'Powered by VitePress',
          copyright: '© 2026 Shen Han',
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
