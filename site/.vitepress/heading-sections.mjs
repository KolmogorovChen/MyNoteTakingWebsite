// Work on parsed block tokens so fences, math and inline markup stay untouched.
export default function headingSections(md) {
  md.core.ruler.push('note_heading_sections', state => {
    if (!state.env.relativePath?.replaceAll('\\', '/').startsWith('notes/')) return;

    const output = [];
    const sections = [];
    const html = content => {
      const token = new state.Token('html_block', '', 0);
      token.content = content + '\n';
      token.block = true;
      output.push(token);
    };
    const close = () => {
      html('</div></details>');
      sections.pop();
    };

    for (const token of state.tokens) {
      // A heading inside a list, quote or custom container cannot outlive it.
      if (token.nesting === -1 && token.type !== 'heading_close') {
        while (sections.length && sections.at(-1).depth > token.level) close();
      }
      if (token.type === 'heading_open') {
        const rank = Number(token.tag.slice(1));
        while (sections.length && sections.at(-1).depth === token.level && sections.at(-1).rank >= rank) close();
        html(`<details class="note-section" data-level="${rank}" open><summary class="note-summary">`);
        sections.push({ rank, depth: token.level });
      }
      output.push(token);
      if (token.type === 'heading_close') html('</summary><div class="note-section-body">');
    }
    while (sections.length) close();
    state.tokens = output;
  });
}
