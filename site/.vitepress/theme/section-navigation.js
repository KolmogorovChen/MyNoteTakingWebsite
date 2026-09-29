import { getScrollOffset, useRoute } from 'vitepress';
import { nextTick, onMounted, onUnmounted, watch } from 'vue';

export function useSectionNavigation() {
  const route = useRoute();
  let frame;

  function reveal(hash) {
    if (!hash) return;
    let id;
    try { id = decodeURIComponent(hash.slice(1)); } catch { return; }
    const target = document.getElementById(id);
    if (!target?.closest('.vp-doc')) return;
    let opened = false;
    for (let section = target.closest('details.note-section'); section; section = section.parentElement?.closest('details.note-section')) {
      if (!section.open) {
        section.open = true;
        opened = true;
      }
    }
    if (opened) {
      // Also correct repeated clicks on the current hash, which emit no hashchange.
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        if (!target.isConnected) return;
        window.scrollTo({ top: window.scrollY + target.getBoundingClientRect().top - getScrollOffset(), behavior: 'instant' });
      });
    }
  }

  const onHashChange = () => reveal(location.hash);
  function onClick(event) {
    // VitePress has already prevented the default in its window capture listener.
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const link = event.target instanceof Element ? event.target.closest('a[href]') : null;
    if (!link || link.hasAttribute('download') || link.hasAttribute('target')) return;
    const url = new URL(link.href, location.href);
    if (url.origin === location.origin && url.pathname === location.pathname && url.search === location.search) reveal(url.hash);
  }

  onMounted(() => {
    document.addEventListener('click', onClick, true);
    window.addEventListener('hashchange', onHashChange);
    onHashChange();
  });
  watch(() => route.path, async () => {
    await nextTick();
    onHashChange();
  });
  onUnmounted(() => {
    document.removeEventListener('click', onClick, true);
    window.removeEventListener('hashchange', onHashChange);
    cancelAnimationFrame(frame);
  });
}
