<script setup lang="ts">
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'
import { data as posts } from './posts.data.js'

// Reverse-chronological index of everything under docs/blog/ that declares a
// `date:`. Built from the content loader rather than hand-maintained, so adding
// a post is a matter of writing it — nothing here needs touching.
const props = defineProps<{ notes?: Record<string, string> }>()
const { lang } = useData()

const items = computed(() =>
  posts.map((p) => ({
    ...p,
    pretty: new Date(p.date).toLocaleDateString(
      lang.value === 'zh-CN' ? 'zh-CN' : 'en-US',
      { year: 'numeric', month: 'long', day: 'numeric', timeZone: 'UTC' },
    ),
    note: props.notes?.[p.url.split('/').pop() ?? ''] ?? p.description,
  })),
)
</script>

<template>
  <ul class="post-list">
    <li v-for="p in items" :key="p.url">
      <a :href="withBase(p.url)">{{ p.title }}</a>
      <time :datetime="p.date">{{ p.pretty }}</time>
      <p>{{ p.note }}</p>
    </li>
  </ul>
</template>

<style scoped>
.post-list { list-style: none; padding: 0; margin: 24px 0 0; }
.post-list li { padding: 20px 0; border-top: 1px solid var(--vp-c-divider); }
.post-list a {
  font-size: 18px; font-weight: 600;
  color: var(--vp-c-brand-1); text-decoration: none;
}
.post-list a:hover { text-decoration: underline; }
.post-list time {
  display: block; margin-top: 4px;
  font-size: 13px; color: var(--vp-c-text-3);
}
.post-list p { margin: 8px 0 0; color: var(--vp-c-text-2); line-height: 1.7; }
</style>
