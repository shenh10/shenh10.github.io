import { createContentLoader } from 'vitepress'

// Every post under docs/blog/ that declares a `date:`, newest first.
// index.md has no date, so it excludes itself.
export default createContentLoader('blog/*.md', {
  transform(raw) {
    return raw
      .filter((p) => p.frontmatter.date)
      .map(({ url, frontmatter }) => ({
        url,
        title: frontmatter.title,
        description: frontmatter.description,
        date: new Date(frontmatter.date).toISOString().slice(0, 10),
      }))
      .sort((a, b) => (a.date < b.date ? 1 : -1))
  },
})
