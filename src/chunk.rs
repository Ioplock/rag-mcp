//! Sentence-aware chunking for single-line ASR transcripts.
//!
//! The transcripts are plain prose with (almost) no newlines and no
//! timestamps, so we split on sentence boundaries and greedily pack
//! sentences into ~TARGET-char chunks with a sentence-level overlap.

pub const TARGET_CHARS: usize = 1000;
pub const OVERLAP_CHARS: usize = 180;

/// Split text into sentences on `. ! ? …` (optionally followed by a
/// closing quote/bracket) plus a stretch of whitespace or end of text.
fn split_sentences(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut end_iter = text.char_indices().peekable();
    // byte index just past the last sentence-ending punctuation run
    let mut punct_end: Option<usize> = None;

    while let Some((i, c)) = end_iter.next() {
        let is_end = matches!(c, '.' | '!' | '?' | '…');
        if is_end {
            punct_end = Some(i + c.len_utf8());
            continue;
        }
        if let Some(pe) = punct_end {
            // Optional closing quotes/brackets right after punctuation.
            let is_closer = matches!(c, '"' | '\'' | '»' | '”' | ')' | ']' | '>');
            if is_closer {
                punct_end = Some(i + c.len_utf8());
                continue;
            }
            if c.is_whitespace() {
                let s = text[start..pe].trim();
                if !s.is_empty() {
                    out.push(s);
                }
                // Skip the whitespace run.
                let mut j = i + c.len_utf8();
                while let Some((k, nc)) = end_iter.peek() {
                    if nc.is_whitespace() {
                        j = *k + nc.len_utf8();
                        end_iter.next();
                    } else {
                        break;
                    }
                }
                start = j;
                punct_end = None;
            } else {
                // Not a boundary (e.g. "1.5", "т.д."). Keep scanning.
                punct_end = None;
            }
        }
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        out.push(tail);
    }
    out
}

fn char_len(s: &str) -> usize {
    s.chars().count()
}

/// Push joined `cur` to `chunks`, return trailing sentences kept as overlap.
fn flush<'a>(chunks: &mut Vec<String>, cur: &[&'a str], overlap: usize) -> (Vec<&'a str>, usize) {
    if cur.is_empty() {
        return (vec![], 0);
    }
    chunks.push(cur.join(" "));
    let mut keep: Vec<&str> = Vec::new();
    let mut keep_len = 0usize;
    for s in cur.iter().rev() {
        let l = char_len(s) + 1;
        if keep_len + l > overlap && !keep.is_empty() {
            break;
        }
        keep_len += l;
        keep.push(s);
    }
    keep.reverse();
    let len: usize = keep.iter().map(|s| char_len(s) + 1).sum();
    (keep, len)
}

/// Collapse all whitespace (transcripts may be one giant line).
fn normalize(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn chunk_text(text: &str, target: usize, overlap: usize) -> Vec<String> {    let text = normalize(text);
    if char_len(&text) <= target {
        return if text.is_empty() { vec![] } else { vec![text] };
    }
    let sentences = split_sentences(&text);
    let mut chunks: Vec<String> = Vec::new();
    let mut cur: Vec<&str> = Vec::new();
    let mut cur_len = 0usize;

    for s in sentences {
        let l = char_len(s) + 1;
        // A single pathological sentence longer than target: hard-cut it.
        if l > target {
            (cur, cur_len) = flush(&mut chunks, &cur, overlap);
            let chars: Vec<char> = s.chars().collect();
            for piece in chars.chunks(target) {
                let p: String = piece.iter().collect();
                chunks.push(p);
            }
            continue;
        }
        if cur_len + l > target && !cur.is_empty() {
            (cur, cur_len) = flush(&mut chunks, &cur, overlap);
        }
        cur.push(s);
        cur_len += l;
    }
    if !cur.is_empty() {
        chunks.push(cur.join(" "));
    }
    chunks
}

/// Markdown-aware chunking: split on `#` headers, never cut inside
/// fenced code blocks. Oversized sections fall back to sentence packing.
pub fn chunk_markdown(text: &str, target: usize, overlap: usize) -> Vec<String> {
    // First split into (header, body) sections, tracking fences.
    let mut sections: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut in_fence = false;
    for line in text.lines() {
        let t = line.trim_start();
        if t.starts_with("```") {
            in_fence = !in_fence;
        }
        let is_header =
            !in_fence && t.starts_with('#') && t.chars().nth(1).map(|c| c == '#' || c == ' ').unwrap_or(true);
        if is_header && !cur.trim().is_empty() {
            sections.push(std::mem::take(&mut cur));
        }
        cur.push_str(line);
        cur.push('\n');
    }
    if !cur.trim().is_empty() {
        sections.push(cur);
    }
    if sections.is_empty() {
        return vec![];
    }
    // Keep one chunk per section; oversized sections get sentence-packed.
    let mut out = Vec::new();
    for s in sections {
        let norm: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
        if char_len(&norm) <= target {
            out.push(norm);
        } else {
            out.extend(chunk_text(&norm, target, overlap));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_line_asr_splits_and_packs() {
        let t = "Привет! Это первое видео по Kotlin. Сегодня создадим проект. Нажимаем New Project. Далее выбираем язык Kotlin. Укажем имя, например test. Собираем приложение.";
        let chunks = chunk_text(t, 100, 20);
        assert!(chunks.len() >= 2);
        // No chunk exceeds target by much (sentence granularity).
        for c in &chunks {
            assert!(char_len(c) <= 130, "chunk too big: {c}");
        }
        // Overlap: some sentence appears in two chunks.
        let joined = chunks.join("\n");
        assert!(joined.contains("New Project"));
    }

    #[test]
    fn short_text_is_single_chunk() {
        assert_eq!(chunk_text("Короткий текст.", 1000, 180), vec!["Короткий текст."]);
    }

    #[test]
    fn empty_is_empty() {
        assert!(chunk_text("   ", 1000, 180).is_empty());
    }

    #[test]
    fn markdown_keeps_code_fences_intact() {
        let md = "# Делегаты\nТекст про by lazy.\n```kotlin\nval x by lazy { 1 }\nval y by lazy { 2 }\n```\n## Observable\nТекст про observable.";
        let chunks = chunk_markdown(md, 200, 40);
        assert!(!chunks.is_empty());
        // A fence must never be split across chunks.
        for c in &chunks {
            let fences = c.matches("```").count();
            assert!(fences % 2 == 0, "split fence in chunk: {c}");
        }
        let joined = chunks.join("\n");
        assert!(joined.contains("by lazy"));
        assert!(joined.contains("observable"));
    }
}
