<script setup lang="ts">
import { computed } from 'vue'
import { useData } from 'vitepress'

// Renders the publication date under a post's title. Only fires on pages that
// declare `date:` in frontmatter, so index and home pages are untouched.
//
// YAML parses an unquoted `date: 2026-08-04` into a Date, but the same field
// arrives as a string once the page data has been serialized — so normalize to
// an ISO string before touching it. Calling a string method on the Date is what
// made an earlier version render nothing: the throw happens inside SSR and is
// swallowed, leaving no output and no build error.
const { frontmatter, lang } = useData()

const iso = computed(() => {
  const d = frontmatter.value.date
  if (!d) return ''
  const parsed = d instanceof Date ? d : new Date(String(d))
  return Number.isNaN(parsed.getTime()) ? '' : parsed.toISOString().slice(0, 10)
})

const published = computed(() => {
  if (!iso.value) return ''
  return new Date(iso.value).toLocaleDateString(
    lang.value === 'zh-CN' ? 'zh-CN' : 'en-US',
    { year: 'numeric', month: 'long', day: 'numeric', timeZone: 'UTC' },
  )
})
</script>

<template>
  <div v-if="published" class="post-meta">
    <time :datetime="iso">{{ published }}</time>
  </div>
</template>

<style scoped>
.post-meta {
  margin: -12px 0 28px;
  font-size: 14px;
  color: var(--vp-c-text-3);
}
</style>
