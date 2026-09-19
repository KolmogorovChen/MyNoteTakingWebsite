<script setup>
import { ref, onMounted } from 'vue';
const props = defineProps({ source: String });
const rendered = ref('');
const error = ref(false);
onMounted(async () => {
  try {
    const { default: mermaid } = await import('mermaid');
    mermaid.initialize({ startOnLoad: false, securityLevel: 'strict', theme: 'neutral' });
    const result = await mermaid.render('diagram-' + Math.random().toString(36).slice(2), decodeURIComponent(props.source));
    rendered.value = result.svg;
  } catch { error.value = true; }
});
</script>
<template>
  <div v-if="rendered" class="mermaid-diagram" v-html="rendered"></div>
  <pre v-else-if="error"><code>{{ decodeURIComponent(source) }}</code></pre>
  <p v-else aria-live="polite">正在绘制流程图…</p>
</template>
