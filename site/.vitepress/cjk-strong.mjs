const cjkLetter = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/u;
const cjkPunctuation = /[\u2018-\u201f\u3001-\u303f\ufe10-\ufe19\ufe30-\ufe6f\uff01-\uff65]/u;

// Keep Markdown's delimiter pairing and escaping; relax only CJK strong boundaries.
export default function cjkStrong(md) {
  const InlineState = md.inline.State;
  const isPunctuation = char => md.utils.isMdAsciiPunct(char.codePointAt(0)) || md.utils.isPunctChar(char);
  const isCjkBoundary = (inside, outside) => isPunctuation(inside)
    && !md.utils.isWhiteSpace(outside.codePointAt(0))
    && (cjkLetter.test(outside) || cjkPunctuation.test(inside));

  // Assign a state class to this renderer, without changing markdown-it's shared prototype.
  md.inline.State = class extends InlineState {
    scanDelims(start, canSplitWord) {
      const scanned = super.scanDelims(start, canSplitWord);
      if (this.src[start] !== '*' || scanned.length < 2) return scanned;
      if (this.cjkStrongDisabled) return scanned;

      // A run with whitespace on both sides cannot be emphasis. Preserve the
      // original pairing for this malformed paragraph, including earlier runs.
      if (!scanned.can_open && !scanned.can_close) {
        this.cjkStrongDisabled = true;
        for (const run of this.cjkStrongRuns || []) {
          for (let i = 0; i < run.length; i++) {
            const delimiter = run.delimiters[run.offset + i];
            delimiter.open = run.open;
            delimiter.close = run.close;
          }
        }
        return scanned;
      }

      const open = scanned.can_open, close = scanned.can_close;
      const before = [...this.src.slice(Math.max(0, start - 2), start)].at(-1) || ' ';
      const after = String.fromCodePoint(this.src.codePointAt(start + scanned.length) ?? 0x20);
      if (isCjkBoundary(after, before)) scanned.can_open = true;
      if (isCjkBoundary(before, after)) scanned.can_close = true;
      if (open !== scanned.can_open || close !== scanned.can_close) {
        (this.cjkStrongRuns ||= []).push({ delimiters: this.delimiters, offset: this.delimiters.length, length: scanned.length, open, close });
      }
      return scanned;
    }
  };
}
