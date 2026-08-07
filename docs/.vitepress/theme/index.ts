import DefaultTheme from 'vitepress/theme'
import MermaidZoom from './MermaidZoom.vue'
import PostMeta from './PostMeta.vue'
import type { Theme } from 'vitepress'
import { h } from 'vue'

export default {
  extends: DefaultTheme,
  Layout() {
    return h(DefaultTheme.Layout, null, {
      // renders the `date:` frontmatter under the post title; a no-op on pages
      // that do not declare one
      'doc-before': () => h(PostMeta),
      'layout-bottom': () => h(MermaidZoom),
    })
  },
} satisfies Theme
