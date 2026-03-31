<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'

const visible = ref(false)
const svgContent = ref('')
const scale = ref(1)

function onClick(e: MouseEvent) {
  const svg = (e.target as Element).closest('.mermaid svg') as SVGElement | null
  if (!svg) return
  svgContent.value = svg.outerHTML
  scale.value = 1.5
  visible.value = true
  document.body.style.overflow = 'hidden'
}

function close() {
  visible.value = false
  document.body.style.overflow = ''
}

function onKeydown(e: KeyboardEvent) {
  if (!visible.value) return
  if (e.key === 'Escape') close()
  if (e.key === '+' || e.key === '=') { scale.value = Math.min(scale.value + 0.25, 5); e.preventDefault() }
  if (e.key === '-') { scale.value = Math.max(scale.value - 0.25, 0.25); e.preventDefault() }
  if (e.key === '0') { scale.value = 1; e.preventDefault() }
}

function onWheel(e: WheelEvent) {
  e.preventDefault()
  const delta = e.deltaY > 0 ? -0.1 : 0.1
  scale.value = Math.min(Math.max(scale.value + delta, 0.25), 5)
}

onMounted(() => {
  document.addEventListener('click', onClick)
  document.addEventListener('keydown', onKeydown)
})

onUnmounted(() => {
  document.removeEventListener('click', onClick)
  document.removeEventListener('keydown', onKeydown)
})
</script>

<template>
  <Teleport to="body">
    <div v-if="visible" class="mermaid-overlay" @wheel.prevent="onWheel">
      <div class="mermaid-toolbar">
        <button @click="scale = Math.max(scale - 0.25, 0.25)">-</button>
        <span class="mermaid-scale">{{ Math.round(scale * 100) }}%</span>
        <button @click="scale = Math.min(scale + 0.25, 5)">+</button>
        <button @click="scale = 1">重置</button>
        <button class="mermaid-close" @click="close">&times;</button>
      </div>
      <div class="mermaid-scroll">
        <div class="mermaid-content" :style="{ transform: `scale(${scale})` }" v-html="svgContent" />
      </div>
    </div>
  </Teleport>
</template>

<style>
.mermaid svg {
  cursor: zoom-in;
}

.mermaid-overlay {
  position: fixed;
  inset: 0;
  z-index: 9999;
  background: var(--vp-c-bg);
  display: flex;
  flex-direction: column;
}

.mermaid-toolbar {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1rem;
  border-bottom: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
  flex-shrink: 0;
}

.mermaid-toolbar button {
  padding: 0.25rem 0.75rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: 4px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
  cursor: pointer;
  font-size: 0.9rem;
}

.mermaid-toolbar button:hover {
  background: var(--vp-c-bg-mute);
}

.mermaid-scale {
  min-width: 3.5rem;
  text-align: center;
  font-size: 0.85rem;
  color: var(--vp-c-text-2);
}

.mermaid-close {
  margin-left: auto !important;
  font-size: 1.3rem !important;
  padding: 0.15rem 0.6rem !important;
}

.mermaid-scroll {
  flex: 1;
  overflow: auto;
  display: flex;
  justify-content: center;
  padding: 2rem;
}

.mermaid-content {
  transform-origin: top center;
  transition: transform 0.15s ease;
}

.mermaid-content svg {
  max-width: none !important;
  height: auto !important;
}
</style>
