//! Post-processing applied to every transcription, all backends.
//!
//! Curly quotes literally break wtype, and a stray newline presses Enter in the
//! target app — in a terminal that *executes* the line. Invisible characters
//! (BOM, zero-width, bidi controls) are worse: they corrupt the injected string
//! without ever showing on screen. All get neutralized.

/// Drop invisible junk, normalize curly quotes to ASCII, collapse newlines to
/// spaces, trim, then apply the user's custom-vocab corrections (proper nouns,
/// jargon, project names the general-English model never learns).
pub fn post_process(s: &str, corrections: &[(String, String)]) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            // Invisible junk — zero-width, BOM, bidi controls, soft hyphen, tag
            // chars. They carry no meaning in dictated text and corrupt the
            // injected string: a BOM mid-line, a hidden bidi override flipping
            // word order, tag chars smuggling ASCII. Dropped, not rewritten.
            _ if is_invisible(c) => {}
            '\u{2018}' | '\u{2019}' | '\u{201B}' | '\u{2032}' => out.push('\''),
            '\u{201C}' | '\u{201D}' | '\u{201F}' | '\u{2033}' => out.push('"'),
            '\n' | '\r' => out.push(' '),
            other => out.push(other),
        }
    }
    apply_corrections(out.trim(), corrections)
}

/// Joins independently transcribed segments without emitting an uncertain
/// boundary word until the following segment can confirm the duplicate.
#[derive(Debug, Default)]
pub struct BoundaryTextJoiner {
    pending_word: Option<String>,
    mark_audio_overlaps: bool,
}

impl BoundaryTextJoiner {
    pub fn with_overlap_markers() -> Self {
        Self {
            pending_word: None,
            mark_audio_overlaps: true,
        }
    }

    pub fn push(
        &mut self,
        text: &str,
        final_segment: bool,
        has_audio_overlap: bool,
    ) -> Option<String> {
        let incoming = text.trim();
        let resolved = match self.pending_word.take() {
            Some(pending) if incoming.is_empty() => return Some(pending),
            Some(pending) => resolve_boundary(
                &pending,
                incoming,
                has_audio_overlap,
                self.mark_audio_overlaps,
            ),
            None => incoming.to_string(),
        };

        if final_segment {
            return (!resolved.is_empty()).then_some(resolved);
        }

        match split_final_word(&resolved) {
            Some((stable, pending)) => {
                self.pending_word = Some(pending.to_string());
                (!stable.is_empty()).then_some(stable.to_string())
            }
            None => (!resolved.is_empty()).then_some(resolved),
        }
    }

    pub fn break_boundary(&mut self) -> Option<String> {
        self.pending_word
            .take()
            .filter(|pending| !pending.is_empty())
    }
}

fn resolve_boundary(
    left: &str,
    right: &str,
    has_audio_overlap: bool,
    mark_audio_overlap: bool,
) -> String {
    let Some((right_word, right_core_end)) = first_word(right) else {
        if has_audio_overlap && mark_audio_overlap {
            return join_with_space(&format!("| {left} |"), right);
        }
        return join_with_space(left, right);
    };
    if !has_audio_overlap {
        return join_with_space(left, right);
    }
    if normalize_word(left) != normalize_word(right_word) {
        return if mark_audio_overlap {
            join_with_space(&format!("| {left} |"), right)
        } else {
            join_with_space(left, right)
        };
    }

    let left_core_end = left
        .char_indices()
        .rfind(|(_, character)| character.is_alphanumeric())
        .map(|(index, character)| index + character.len_utf8())
        .unwrap_or(left.len());
    if mark_audio_overlap {
        return format!("| {} |{}", &left[..left_core_end], &right[right_core_end..]);
    }
    let mut joined = String::with_capacity(left_core_end + right.len() - right_core_end);
    joined.push_str(&left[..left_core_end]);
    joined.push_str(&right[right_core_end..]);
    joined
}

fn first_word(text: &str) -> Option<(&str, usize)> {
    let core_start = text
        .char_indices()
        .find(|(_, character)| character.is_alphanumeric())?
        .0;
    let token_end = text[core_start..]
        .char_indices()
        .find(|(_, character)| character.is_whitespace())
        .map(|(index, _)| core_start + index)
        .unwrap_or(text.len());
    let core_end = text[core_start..token_end]
        .char_indices()
        .rfind(|(_, character)| character.is_alphanumeric())
        .map(|(index, character)| core_start + index + character.len_utf8())?;
    Some((&text[core_start..core_end], core_end))
}

fn normalize_word(word: &str) -> String {
    word.chars()
        .filter(|character| character.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn split_final_word(text: &str) -> Option<(&str, &str)> {
    let trimmed = text.trim_end();
    let final_word = trimmed
        .char_indices()
        .rfind(|(_, character)| character.is_alphanumeric())?
        .0;
    let token_start = trimmed[..final_word]
        .char_indices()
        .rfind(|(_, character)| character.is_whitespace())
        .map(|(index, character)| index + character.len_utf8())
        .unwrap_or(0);
    Some((trimmed[..token_start].trim_end(), &trimmed[token_start..]))
}

fn join_with_space(left: &str, right: &str) -> String {
    match (left.is_empty(), right.is_empty()) {
        (true, _) => right.to_string(),
        (_, true) => left.to_string(),
        (false, false) => format!("{left} {right}"),
    }
}

/// Zero-width, byte-order-mark, bidirectional-control, soft-hyphen and tag
/// characters: rendered as nothing, yet able to break injection or hide intent.
/// Visible content (accents, combining marks, CJK, emoji) is deliberately *not*
/// matched — this strips junk, it does not edit the model's words. In
/// particular ZWNJ/ZWJ (U+200C/U+200D) are kept: they bind emoji sequences and
/// are orthographically required in some scripts, so they carry meaning.
fn is_invisible(c: char) -> bool {
    matches!(c,
        '\u{00AD}'                  // soft hyphen
        | '\u{061C}'                // arabic letter mark
        | '\u{180E}'                // mongolian vowel separator
        | '\u{200B}'                // zero-width space
        | '\u{200E}'..='\u{200F}'   // LRM, RLM (bidi marks) — note: 200C/200D skipped
        | '\u{202A}'..='\u{202E}'   // bidi embeddings & overrides
        | '\u{2060}'..='\u{2064}'   // word joiner, invisible operators
        | '\u{2066}'..='\u{206F}'   // bidi isolates + deprecated format chars
        | '\u{FEFF}'                // BOM / zero-width no-break space
        | '\u{FFF9}'..='\u{FFFB}'   // interlinear annotation anchors
        | '\u{E0000}'..='\u{E007F}' // tag characters (hidden-ASCII smuggling)
    )
}

/// Whole-word, case-insensitive find-and-replace. Patterns are matched
/// longest-first so a more specific phrase wins over a shorter prefix, and only
/// on word boundaries (alphanumeric runs) so "can" never fires inside "candle".
/// No regex dep — a hand-rolled boundary scan.
fn apply_corrections(s: &str, corrections: &[(String, String)]) -> String {
    if corrections.is_empty() {
        return s.to_string();
    }
    let mut rules: Vec<(&str, &str)> = corrections
        .iter()
        .filter(|(from, _)| !from.is_empty())
        .map(|(from, to)| (from.as_str(), to.as_str()))
        .collect();
    rules.sort_by_key(|r| std::cmp::Reverse(r.0.chars().count()));

    let lower = s.to_lowercase();
    let chars: Vec<char> = s.chars().collect();
    let lower_chars: Vec<char> = lower.chars().collect();
    // `to_lowercase` is not 1:1 for all scripts; bail to a no-op rather than
    // misalign indices on the rare char whose lowercase widens.
    if lower_chars.len() != chars.len() {
        return s.to_string();
    }

    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        let at_start = i == 0 || !chars[i - 1].is_alphanumeric();
        let mut matched = false;
        if at_start {
            for (from, to) in &rules {
                let pat: Vec<char> = from.to_lowercase().chars().collect();
                let end = i + pat.len();
                if end <= lower_chars.len()
                    && lower_chars[i..end] == pat[..]
                    && (end == chars.len() || !chars[end].is_alphanumeric())
                {
                    out.push_str(to);
                    i = end;
                    matched = true;
                    break;
                }
            }
        }
        if !matched {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{post_process, BoundaryTextJoiner};

    fn pp(s: &str) -> String {
        post_process(s, &[])
    }

    fn rules(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect()
    }

    #[test]
    fn trims_whitespace() {
        assert_eq!(pp("  hello  "), "hello");
    }

    #[test]
    fn normalizes_curly_quotes() {
        assert_eq!(pp("\u{2018}hi\u{2019}"), "'hi'");
        assert_eq!(pp("\u{201C}hi\u{201D}"), "\"hi\"");
        assert_eq!(pp("it\u{2019}s"), "it's");
    }

    #[test]
    fn collapses_newlines() {
        assert_eq!(pp("a\nb"), "a b");
        assert_eq!(pp("ls\n"), "ls");
    }

    #[test]
    fn strips_invisible_and_bom() {
        assert_eq!(pp("\u{FEFF}hello"), "hello"); // BOM
        assert_eq!(pp("hel\u{200B}lo"), "hello"); // zero-width space
        assert_eq!(pp("soft\u{00AD}hyphen"), "softhyphen");
        assert_eq!(pp("a\u{202E}b"), "ab"); // right-to-left override
        assert_eq!(pp("x\u{2060}y"), "xy"); // word joiner
        assert_eq!(pp("l\u{200E}r"), "lr"); // left-to-right mark
        assert_eq!(pp("tag\u{E0041}end"), "tagend"); // tag character
                                                     // A BOM that would otherwise survive the trim and corrupt injection.
        assert_eq!(pp("  \u{FEFF}ls "), "ls");
    }

    #[test]
    fn preserves_visible_unicode() {
        // Accents, combining marks, CJK and emoji are real content — kept as-is.
        assert_eq!(pp("café"), "café");
        assert_eq!(pp("e\u{0301}"), "e\u{0301}"); // combining acute is visible
        assert_eq!(pp("日本語"), "日本語");
        assert_eq!(pp("emoji 😀 ok"), "emoji 😀 ok");
        // ZWJ binds an emoji sequence into one glyph; ZWNJ is orthographic. Both
        // carry meaning, so they survive — they are not "invisible junk".
        let family = "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}";
        assert_eq!(pp(family), family);
        assert_eq!(pp("\u{200C}"), "\u{200C}"); // ZWNJ kept
    }

    #[test]
    fn corrections_case_insensitive() {
        let r = rules(&[("git hub", "GitHub"), ("claude", "Claude")]);
        assert_eq!(post_process("push to Git Hub", &r), "push to GitHub");
        assert_eq!(post_process("ask CLAUDE", &r), "ask Claude");
    }

    #[test]
    fn corrections_respect_word_boundaries() {
        let r = rules(&[("can", "CAN")]);
        // mid-word "can" inside "candle"/"scan" must not fire
        assert_eq!(
            post_process("a candle I can scan", &r),
            "a candle I CAN scan"
        );
    }

    #[test]
    fn corrections_longest_match_first() {
        // a longer, more specific phrase wins over a shorter prefix rule
        let r = rules(&[("new", "NEW"), ("new york", "New York")]);
        assert_eq!(post_process("new york is new", &r), "New York is NEW");
    }

    #[test]
    fn corrections_multi_word_pattern() {
        let r = rules(&[("my voice", "my-voice")]);
        assert_eq!(
            post_process("I use my voice daily", &r),
            "I use my-voice daily"
        );
    }

    #[test]
    fn empty_corrections_is_noop() {
        assert_eq!(post_process("git hub", &[]), "git hub");
    }

    #[test]
    fn boundary_join_keeps_left_word_and_right_punctuation() {
        let mut joiner = BoundaryTextJoiner::default();
        assert_eq!(
            joiner.push("Hello world.", false, false).as_deref(),
            Some("Hello")
        );
        assert_eq!(
            joiner.push("World, this works.", true, true).as_deref(),
            Some("world, this works.")
        );
    }

    #[test]
    fn boundary_join_keeps_exact_left_spelling() {
        let mut joiner = BoundaryTextJoiner::default();
        assert_eq!(
            joiner.push("So happy. Someone...", false, false).as_deref(),
            Some("So happy.")
        );
        assert_eq!(
            joiner.push("someone helped", true, true).as_deref(),
            Some("Someone helped")
        );
    }

    #[test]
    fn diagnostic_join_marks_each_confirmed_audio_overlap() {
        let mut joiner = BoundaryTextJoiner::with_overlap_markers();
        let mut text = String::new();
        for chunk in [
            joiner.push("So happy. Someone...", false, false),
            joiner.push("someone helped another", false, true),
            joiner.push("Another person arrived.", true, true),
        ]
        .into_iter()
        .flatten()
        {
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(&chunk);
        }
        assert_eq!(
            text,
            "So happy. | Someone | helped | another | person arrived."
        );
    }

    #[test]
    fn diagnostic_join_marks_an_overlap_mismatch_without_merging_it() {
        let mut joiner = BoundaryTextJoiner::with_overlap_markers();
        assert_eq!(
            joiner.push("hello world.", false, false).as_deref(),
            Some("hello")
        );
        assert_eq!(
            joiner.push("Different start.", true, true).as_deref(),
            Some("| world. | Different start.")
        );
    }

    #[test]
    fn boundary_join_ignores_leading_punctuation_and_capitalization() {
        let mut joiner = BoundaryTextJoiner::default();
        assert_eq!(
            joiner.push("say hello!", false, false).as_deref(),
            Some("say")
        );
        assert_eq!(
            joiner.push("... HELLO? Again", true, true).as_deref(),
            Some("hello? Again")
        );
    }

    #[test]
    fn boundary_join_preserves_both_words_on_mismatch() {
        let mut joiner = BoundaryTextJoiner::default();
        assert_eq!(
            joiner.push("hello world.", false, false).as_deref(),
            Some("hello")
        );
        assert_eq!(
            joiner.push("Different start.", true, true).as_deref(),
            Some("world. Different start.")
        );
    }

    #[test]
    fn release_flushes_an_unconfirmed_boundary_word() {
        let mut joiner = BoundaryTextJoiner::default();
        assert_eq!(
            joiner.push("only word.", false, false).as_deref(),
            Some("only")
        );
        assert_eq!(joiner.push("", true, false).as_deref(), Some("word."));
        assert_eq!(joiner.break_boundary(), None);
    }

    #[test]
    fn equal_words_without_audio_overlap_are_both_kept() {
        let mut joiner = BoundaryTextJoiner::default();
        assert_eq!(joiner.push("very", false, false), None);
        assert_eq!(
            joiner.push("Very good", true, false).as_deref(),
            Some("very Very good")
        );
    }

    #[test]
    fn empty_transcript_breaks_the_pending_boundary() {
        let mut joiner = BoundaryTextJoiner::default();
        assert_eq!(
            joiner.push("first word", false, false).as_deref(),
            Some("first")
        );
        assert_eq!(joiner.push("", false, true).as_deref(), Some("word"));
        assert_eq!(
            joiner.push("Word again", true, true).as_deref(),
            Some("Word again")
        );
    }

    #[test]
    fn failed_transcript_breaks_the_pending_boundary() {
        let mut joiner = BoundaryTextJoiner::default();
        assert_eq!(
            joiner.push("first word", false, false).as_deref(),
            Some("first")
        );
        assert_eq!(joiner.break_boundary().as_deref(), Some("word"));
        assert_eq!(
            joiner.push("Word again", true, true).as_deref(),
            Some("Word again")
        );
    }
}
