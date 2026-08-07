<script setup lang="ts">
// Two-column profile layout: a sticky portrait and compact sidebar on the
// left, name / title / body on the right. Deliberately not VitePress's home
// hero, whose 56px gradient heading is a landing-page device — a profile
// wants the name at heading size and the affiliation right under it.
defineProps<{
  name: string
  title: string
  photo: string
  caption?: string
  interestsLabel?: string
  interests?: string[]
  educationLabel?: string
  education?: { degree: string; school: string; year: string }[]
  links?: { text: string; href: string }[]
}>()
</script>

<template>
  <div class="profile">
    <aside class="profile-side">
      <img class="profile-photo" :src="photo" :alt="name" />
      <p v-if="caption" class="profile-caption">{{ caption }}</p>

      <div v-if="interests?.length" class="side-block">
        <h2 class="side-title">{{ interestsLabel ?? 'Research Interests' }}</h2>
        <ul class="side-tags">
          <li v-for="i in interests" :key="i">{{ i }}</li>
        </ul>
      </div>

      <div v-if="education?.length" class="side-block">
        <h2 class="side-title">{{ educationLabel ?? 'Education' }}</h2>
        <ul class="side-edu">
          <li v-for="e in education" :key="e.degree + e.year">
            <span class="edu-degree">{{ e.degree }}</span>
            <span class="edu-school">{{ e.school }}</span>
            <span class="edu-year">{{ e.year }}</span>
          </li>
        </ul>
      </div>
    </aside>

    <main class="profile-main">
      <h1 class="profile-name">{{ name }}</h1>
      <p class="profile-title">{{ title }}</p>
      <p v-if="links?.length" class="profile-links">
        <template v-for="(l, n) in links" :key="l.href">
          <span v-if="n" class="sep">·</span>
          <a :href="l.href" target="_blank" rel="noreferrer">{{ l.text }}</a>
        </template>
      </p>
      <div class="vp-doc profile-body"><slot /></div>
    </main>
  </div>
</template>

<style scoped>
.profile {
  max-width: 1100px;
  margin: 0 auto;
  padding: 48px 24px 96px;
  display: grid;
  grid-template-columns: 280px 1fr;
  gap: 56px;
  align-items: start;
}

.profile-side { position: sticky; top: 96px; }

.profile-photo {
  width: 100%;
  border-radius: 8px;
  border: 3px solid var(--vp-c-bg-soft);
  box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
}

.profile-caption {
  margin-top: 10px;
  font-size: 12.5px;
  line-height: 1.4;
  color: var(--vp-c-text-3);
  text-align: center;
}

.side-block {
  margin-top: 28px;
  padding-top: 24px;
  border-top: 1px solid var(--vp-c-divider);
}

.side-title {
  margin: 0 0 12px;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--vp-c-text-1);
  border: 0;
  padding: 0;
}

.side-tags { list-style: none; padding: 0; margin: 0; display: flex; flex-wrap: wrap; gap: 6px; }
.side-tags li {
  font-size: 12.5px;
  padding: 3px 9px;
  border-radius: 12px;
  background: var(--vp-c-default-soft);
  color: var(--vp-c-text-2);
}

.side-edu { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 14px; }
.side-edu li { font-size: 13px; line-height: 1.45; }
.edu-degree { display: block; font-weight: 600; color: var(--vp-c-text-1); }
.edu-school, .edu-year { display: block; color: var(--vp-c-text-3); }

.profile-name {
  margin: 0;
  font-size: 36px;
  font-weight: 700;
  line-height: 1.2;
  letter-spacing: -0.02em;
  color: var(--vp-c-text-1);
}

.profile-title {
  margin: 8px 0 0;
  font-size: 17.5px;
  font-weight: 500;
  color: var(--vp-c-text-2);
}

.profile-links { margin: 12px 0 0; font-size: 14px; }
.profile-links a { color: var(--vp-c-brand-1); text-decoration: none; }
.profile-links a:hover { text-decoration: underline; }
.profile-links .sep { margin: 0 8px; color: var(--vp-c-text-3); }

/* the first section heading sits right under the header — no rule above it */
.profile-body :deep(h2:first-of-type) { margin-top: 32px; border-top: 0; padding-top: 0; }

@media (max-width: 860px) {
  .profile { grid-template-columns: 1fr; gap: 32px; padding: 32px 24px 64px; }
  .profile-side { position: static; }
  .profile-photo { max-width: 200px; display: block; }
  .profile-caption { text-align: left; max-width: 200px; }
  .profile-name { font-size: 30px; }
}
</style>
