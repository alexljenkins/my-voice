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
    use super::post_process;

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
}
