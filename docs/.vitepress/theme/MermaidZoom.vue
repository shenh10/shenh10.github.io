<script setup lang="ts">
import { onMounted, onUnmounted, ref, nextTick } from 'vue'

const visible = ref(false)
const svgHtml = ref('')
const scale = ref(1)
const contentRef = ref<HTMLElement | null>(null)
const dragging = ref(false)
const dragStart = { x: 0, y: 0, scrollX: 0, scrollY: 0 }
const copyTip = ref('')

function onClick(e: MouseEvent) {
  const svg = (e.target as Element).closest('.mermaid svg') as SVGElement | null
  if (!svg) return

  const clone = svg.cloneNode(true) as SVGElement
  const w = svg.getAttribute('width') || svg.getBoundingClientRect().width.toString()
  const h = svg.getAttribute('height') || svg.getBoundingClientRect().height.toString()
  const vb = svg.getAttribute('viewBox') || `0 0 ${parseFloat(w)} ${parseFloat(h)}`

  clone.removeAttribute('width')
  clone.removeAttribute('height')
  clone.removeAttribute('style')
  clone.setAttribute('viewBox', vb)

  svgHtml.value = clone.outerHTML
  scale.value = 1
  visible.value = true
  document.body.style.overflow = 'hidden'

  nextTick(() => {
    contentRef.value?.scrollTo(0, 0)
  })
}

function close() {
  visible.value = false
  document.body.style.overflow = ''
}

function zoomIn() { scale.value = Math.min(scale.value + 0.25, 5) }
function zoomOut() { scale.value = Math.max(scale.value - 0.25, 0.5) }
function zoomReset() { scale.value = 1 }

function onKeydown(e: KeyboardEvent) {
  if (!visible.value) return
  if (e.key === 'Escape') close()
  if (e.key === '+' || e.key === '=') { zoomIn(); e.preventDefault() }
  if (e.key === '-') { zoomOut(); e.preventDefault() }
  if (e.key === '0') { zoomReset(); e.preventDefault() }
}

function onWheel(e: WheelEvent) {
  e.preventDefault()
  if (e.deltaY > 0) zoomOut()
  else zoomIn()
}

async function copyAsImage(ratio: number) {
  const svgEl = contentRef.value?.querySelector('svg')
  if (!svgEl) return

  try {
    const pngPromise = (async () => {
      const clone = svgEl.cloneNode(true) as SVGElement
      const vb = clone.getAttribute('viewBox')?.split(' ').map(Number) || [0, 0, 800, 600]
      const w = Math.round(vb[2] * ratio)
      const h = Math.round(vb[3] * ratio)
      clone.setAttribute('width', String(w))
      clone.setAttribute('height', String(h))
      clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')

      const data = new XMLSerializer().serializeToString(clone)
      const svgDataUrl = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(data)))

      const img = await new Promise<HTMLImageElement>((resolve, reject) => {
        const i = new Image()
        i.onload = () => resolve(i)
        i.onerror = reject
        i.src = svgDataUrl
      })

      const canvas = document.createElement('canvas')
      canvas.width = w
      canvas.height = h
      const ctx = canvas.getContext('2d')!
      ctx.fillStyle = '#ffffff'
      ctx.fillRect(0, 0, w, h)
      ctx.drawImage(img, 0, 0)

      return await new Promise<Blob>((resolve, reject) =>
        canvas.toBlob((b) => b ? resolve(b) : reject(), 'image/png')
      )
    })()

    // Pass promise directly to ClipboardItem to preserve user gesture
    await navigator.clipboard.write([
      new ClipboardItem({ 'image/png': pngPromise })
    ])
    copyTip.value = '已复制'
  } catch {
    copyTip.value = '复制失败'
  }
  setTimeout(() => { copyTip.value = '' }, 2000)
}

function onPointerDown(e: PointerEvent) {
  const el = contentRef.value
  if (!el) return
  dragging.value = true
  dragStart.x = e.clientX
  dragStart.y = e.clientY
  dragStart.scrollX = el.scrollLeft
  dragStart.scrollY = el.scrollTop
  ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
}

function onPointerMove(e: PointerEvent) {
  if (!dragging.value) return
  const el = contentRef.value
  if (!el) return
  el.scrollLeft = dragStart.scrollX - (e.clientX - dragStart.x)
  el.scrollTop = dragStart.scrollY - (e.clientY - dragStart.y)
}

function onPointerUp() {
  dragging.value = false
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
    <div v-if="visible" class="mz-overlay" @wheel.prevent="onWheel">
      <div class="mz-toolbar">
        <button @click="zoomOut">-</button>
        <span class="mz-scale">{{ Math.round(scale * 100) }}%</span>
        <button @click="zoomIn">+</button>
        <button @click="zoomReset">重置</button>
        <button @click="copyAsImage(scale)">复制当前尺寸</button>
        <span v-if="copyTip" class="mz-tip">{{ copyTip }}</span>
        <button class="mz-close" @click="close">&times;</button>
      </div>
      <div ref="contentRef" class="mz-body" :class="{ 'mz-grabbing': dragging }"
           @pointerdown="onPointerDown" @pointermove="onPointerMove" @pointerup="onPointerUp" @pointercancel="onPointerUp">
        <div class="mz-svg-wrap" :style="{ width: (scale * 100) + '%' }" v-html="svgHtml" />
      </div>
    </div>
  </Teleport>
</template>

<style>
.mermaid svg {
  cursor: zoom-in;
}

.mz-overlay {
  position: fixed;
  inset: 0;
  z-index: 9999;
  background: var(--vp-c-bg);
  display: flex;
  flex-direction: column;
}

.mz-toolbar {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  border-bottom: 1px solid var(--vp-c-divider);
  flex-shrink: 0;
}

.mz-toolbar button {
  padding: 0.25rem 0.75rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: 4px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
  cursor: pointer;
  font-size: 0.9rem;
}

.mz-toolbar button:hover {
  background: var(--vp-c-bg-mute);
}

.mz-scale {
  min-width: 3.5rem;
  text-align: center;
  font-size: 0.85rem;
  color: var(--vp-c-text-2);
}

.mz-tip {
  font-size: 0.85rem;
  color: var(--vp-c-brand);
}

.mz-close {
  margin-left: auto !important;
  font-size: 1.5rem !important;
  padding: 0.1rem 0.6rem !important;
}

.mz-body {
  flex: 1;
  overflow: auto;
  padding: 1rem;
  cursor: grab;
  user-select: none;
}

.mz-body.mz-grabbing {
  cursor: grabbing;
}

.mz-svg-wrap {
  margin: 0 auto;
  transition: width 0.15s ease;
}

.mz-svg-wrap svg {
  width: 100% !important;
  height: auto !important;
  max-width: none !important;
}
</style>
