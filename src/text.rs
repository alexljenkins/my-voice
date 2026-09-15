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

/// Joins independently transcribed segments while keeping their shared audio
/// from producing duplicate words at each boundary.
#[derive(Debug, Default)]
pub struct BoundaryTextJoiner {
    pending_punctuation: String,
    trailing_words: Vec<String>,
}

impl BoundaryTextJoiner {
    pub fn push(
        &mut self,
        text: &str,
        final_segment: bool,
        has_audio_overlap: bool,
    ) -> Option<String> {
        let incoming = text.trim();
        if incoming.is_empty() {
            return final_segment
                .then(|| std::mem::take(&mut self.pending_punctuation))
                .filter(|punctuation| !punctuation.is_empty());
        }

        let words = word_spans(incoming);
        let dropped_words =
            if has_audio_overlap && !self.trailing_words.is_empty() && !words.is_empty() {
                aligned_prefix_len(&self.trailing_words, &words).max(1)
            } else {
                0
            };

        let mut resolved = if dropped_words == 0 {
            join_boundary(&std::mem::take(&mut self.pending_punctuation), incoming)
        } else {
            self.pending_punctuation.clear();
            let dropped = &words[dropped_words - 1];
            incoming[dropped.core_end..].trim_start().to_string()
        };

        remember_trailing_words(&mut self.trailing_words, &words[dropped_words..]);

        if final_segment {
            return (!resolved.is_empty()).then_some(resolved);
        }

        if dropped_words == words.len() && dropped_words > 0 {
            self.pending_punctuation = resolved;
            return None;
        }

        let (body, punctuation) = strip_final_punctuation(&resolved);
        resolved.truncate(body);
        self.pending_punctuation = punctuation;
        (!resolved.is_empty()).then_some(resolved)
    }

    pub fn break_boundary(&mut self) -> Option<String> {
        self.trailing_words.clear();
        (!self.pending_punctuation.is_empty())
            .then(|| std::mem::take(&mut self.pending_punctuation))
    }
}

#[cfg(any(test, feature = "debug-tools"))]
#[derive(Debug, Default)]
pub struct BoundaryTextDiagnostic {
    pending_segment: Option<String>,
}

#[cfg(any(test, feature = "debug-tools"))]
impl BoundaryTextDiagnostic {
    pub fn push(
        &mut self,
        text: &str,
        final_segment: bool,
        has_audio_overlap: bool,
    ) -> Option<String> {
        let incoming = text.trim();
        if incoming.is_empty() {
            return final_segment
                .then(|| self.pending_segment.take())
                .flatten()
                .filter(|pending| !pending.is_empty());
        }

        let Some(previous) = self.pending_segment.take() else {
            if final_segment {
                return Some(incoming.to_string());
            }
            self.pending_segment = Some(incoming.to_string());
            return None;
        };

        let previous_words = word_spans(&previous);
        let incoming_words = word_spans(incoming);
        if !has_audio_overlap || previous_words.is_empty() || incoming_words.is_empty() {
            if final_segment {
                return Some(join_with_space(&previous, incoming));
            }
            self.pending_segment = Some(incoming.to_string());
            return Some(previous);
        }

        let trailing: Vec<String> = previous_words
            .iter()
            .map(|word| word.normalized.clone())
            .collect();
        let overlap_words = aligned_prefix_len(&trailing, &incoming_words).max(1);
        let previous_overlap = &previous_words[previous_words.len() - overlap_words];
        let incoming_overlap = &incoming_words[overlap_words - 1];
        let before = previous[..previous_overlap.token_start].trim_end();
        let old = previous[previous_overlap.token_start..].trim();
        let new = incoming[..incoming_overlap.token_end].trim();
        let novel = incoming[incoming_overlap.token_end..].trim_start();
        let compared = format_diagnostic_boundary(before, old, new);

        if final_segment {
            return Some(join_with_space(&compared, novel));
        }
        self.pending_segment = Some(novel.to_string());
        Some(compared)
    }

    pub fn break_boundary(&mut self) -> Option<String> {
        self.pending_segment
            .take()
            .filter(|pending| !pending.is_empty())
    }
}

#[derive(Debug)]
struct WordSpan {
    normalized: String,
    #[cfg(any(test, feature = "debug-tools"))]
    token_start: usize,
    core_end: usize,
    #[cfg(any(test, feature = "debug-tools"))]
    token_end: usize,
}

fn word_spans(text: &str) -> Vec<WordSpan> {
    let mut words = Vec::new();
    let mut token_start = None;
    for (index, character) in text
        .char_indices()
        .chain(std::iter::once((text.len(), ' ')))
    {
        if character.is_whitespace() {
            if let Some(start) = token_start.take() {
                let token = &text[start..index];
                if let Some((core_end, normalized)) = normalized_token(token, start) {
                    words.push(WordSpan {
                        normalized,
                        #[cfg(any(test, feature = "debug-tools"))]
                        token_start: start,
                        core_end,
                        #[cfg(any(test, feature = "debug-tools"))]
                        token_end: index,
                    });
                }
            }
        } else if token_start.is_none() {
            token_start = Some(index);
        }
    }
    words
}

#[cfg(any(test, feature = "debug-tools"))]
fn format_diagnostic_boundary(before: &str, old: &str, new: &str) -> String {
    let mut output = String::new();
    if !before.is_empty() {
        output.push_str(before);
        output.push(' ');
    }
    output.push_str("| ");
    output.push_str(old);
    output.push_str(" || ");
    output.push_str(new);
    output.push_str(" |");
    output
}

fn normalized_token(token: &str, offset: usize) -> Option<(usize, String)> {
    let core_end = token
        .char_indices()
        .rfind(|(_, character)| character.is_alphanumeric())
        .map(|(index, character)| offset + index + character.len_utf8())?;
    let normalized = token
        .chars()
        .filter(|character| character.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect();
    Some((core_end, normalized))
}

fn aligned_prefix_len(trailing: &[String], incoming: &[WordSpan]) -> usize {
    let max_words = trailing.len().min(incoming.len()).min(2);
    (1..=max_words)
        .rev()
        .find(|&count| {
            let left = &trailing[trailing.len() - count..];
            let right = &incoming[..count];
            let distance = left
                .iter()
                .zip(right)
                .map(|(left, right)| edit_distance(left, &right.normalized))
                .sum::<usize>();
            let length = left
                .iter()
                .zip(right)
                .map(|(left, right)| left.chars().count().max(right.normalized.chars().count()))
                .sum::<usize>();
            distance == 0 || distance.saturating_mul(4) <= length
        })
        .unwrap_or(0)
}

fn edit_distance(left: &str, right: &str) -> usize {
    let right: Vec<char> = right.chars().collect();
    let mut row: Vec<usize> = (0..=right.len()).collect();
    for (left_index, left_character) in left.chars().enumerate() {
        let mut diagonal = row[0];
        row[0] = left_index + 1;
        for (right_index, right_character) in right.iter().enumerate() {
            let above = row[right_index + 1];
            row[right_index + 1] = if left_character == *right_character {
                diagonal
            } else {
                1 + diagonal.min(above).min(row[right_index])
            };
            diagonal = above;
        }
    }
    row[right.len()]
}

fn remember_trailing_words(trailing: &mut Vec<String>, words: &[WordSpan]) {
    trailing.extend(words.iter().map(|word| word.normalized.clone()));
    if trailing.len() > 2 {
        trailing.drain(..trailing.len() - 2);
    }
}

fn strip_final_punctuation(text: &str) -> (usize, String) {
    let Some((index, character)) = text
        .char_indices()
        .rfind(|(_, character)| character.is_alphanumeric())
    else {
        return (text.len(), String::new());
    };
    let core_end = index + character.len_utf8();
    (core_end, text[core_end..].to_string())
}

fn join_boundary(punctuation: &str, text: &str) -> String {
    if punctuation.is_empty() {
        text.to_string()
    } else if text.is_empty() {
        punctuation.to_string()
    } else {
        format!("{punctuation} {text}")
    }
}

#[cfg(any(test, feature = "debug-tools"))]
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
    use super::{post_process, BoundaryTextDiagnostic, BoundaryTextJoiner};

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
    fn nonfinal_segment_emits_its_last_word_without_punctuation() {
        let mut joiner = BoundaryTextJoiner::default();
        assert_eq!(
            joiner.push("Hello world.", false, false).as_deref(),
            Some("Hello world")
        );
        assert_eq!(joiner.break_boundary().as_deref(), Some("."));
    }

    #[test]
    fn boundary_join_keeps_left_word_and_new_punctuation() {
        let mut joiner = BoundaryTextJoiner::default();
        assert_eq!(
            joiner.push("Hello world.", false, false).as_deref(),
            Some("Hello world")
        );
        assert_eq!(
            joiner.push("World, this works.", true, true).as_deref(),
            Some(", this works.")
        );
    }

    #[test]
    fn boundary_join_keeps_exact_left_spelling() {
        let mut joiner = BoundaryTextJoiner::default();
        assert_eq!(
            joiner.push("So happy. Someone...", false, false).as_deref(),
            Some("So happy. Someone")
        );
        assert_eq!(
            joiner.push("someone helped", true, true).as_deref(),
            Some("helped")
        );
    }

    #[test]
    fn diagnostic_join_marks_each_confirmed_audio_overlap() {
        let mut joiner = BoundaryTextDiagnostic::default();
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
            "So happy. | Someone... || someone | helped | another || Another | person arrived."
        );
    }

    #[test]
    fn diagnostic_join_marks_an_overlap_mismatch_without_merging_it() {
        let mut joiner = BoundaryTextDiagnostic::default();
        assert_eq!(joiner.push("hello world.", false, false), None);
        assert_eq!(
            joiner.push("Different start.", true, true).as_deref(),
            Some("hello | world. || Different | start.")
        );
    }

    #[test]
    fn diagnostic_join_marks_both_sides_of_a_two_word_alignment() {
        let mut joiner = BoundaryTextDiagnostic::default();
        assert_eq!(joiner.push("we can monetize it.", false, false), None);
        assert_eq!(
            joiner
                .push("Monetise it, then grow.", true, true)
                .as_deref(),
            Some("we can | monetize it. || Monetise it, | then grow.")
        );
    }

    #[test]
    fn diagnostic_join_flushes_a_segment_when_the_boundary_breaks() {
        let mut joiner = BoundaryTextDiagnostic::default();
        assert_eq!(joiner.push("hello world.", false, false), None);
        assert_eq!(joiner.break_boundary().as_deref(), Some("hello world."));
    }

    #[test]
    fn boundary_join_ignores_leading_punctuation_and_capitalization() {
        let mut joiner = BoundaryTextJoiner::default();
        assert_eq!(
            joiner.push("say hello!", false, false).as_deref(),
            Some("say hello")
        );
        assert_eq!(
            joiner.push("... HELLO? Again", true, true).as_deref(),
            Some("? Again")
        );
    }

    #[test]
    fn boundary_join_always_drops_the_first_overlap_word_on_mismatch() {
        let mut joiner = BoundaryTextJoiner::default();
        assert_eq!(
            joiner.push("hello world.", false, false).as_deref(),
            Some("hello world")
        );
        assert_eq!(
            joiner.push("Different start.", true, true).as_deref(),
            Some("start.")
        );
    }

    #[test]
    fn boundary_join_keeps_punctuation_after_a_mismatched_overlap_word() {
        let mut joiner = BoundaryTextJoiner::default();
        assert_eq!(
            joiner.push("hello world.", false, false).as_deref(),
            Some("hello world")
        );
        assert_eq!(
            joiner.push("Different, start.", true, true).as_deref(),
            Some(", start.")
        );
    }

    #[test]
    fn boundary_join_uses_a_two_word_alignment() {
        let mut joiner = BoundaryTextJoiner::default();
        assert_eq!(
            joiner.push("we can monetize it.", false, false).as_deref(),
            Some("we can monetize it")
        );
        assert_eq!(
            joiner
                .push("Monetise it, then grow.", true, true)
                .as_deref(),
            Some(", then grow.")
        );
    }

    #[test]
    fn release_flushes_the_held_punctuation() {
        let mut joiner = BoundaryTextJoiner::default();
        assert_eq!(
            joiner.push("only word.", false, false).as_deref(),
            Some("only word")
        );
        assert_eq!(joiner.push("", true, false).as_deref(), Some("."));
        assert_eq!(joiner.break_boundary(), None);
    }

    #[test]
    fn overlap_only_segment_holds_its_punctuation() {
        let mut joiner = BoundaryTextJoiner::default();
        assert_eq!(joiner.push("hello", false, false).as_deref(), Some("hello"));
        assert_eq!(joiner.push("Hello.", false, true), None);
        assert_eq!(joiner.push("Hello!", true, true).as_deref(), Some("!"));
    }

    #[test]
    fn a_segment_without_audio_overlap_keeps_both_words() {
        let mut joiner = BoundaryTextJoiner::default();
        assert_eq!(joiner.push("very.", false, false).as_deref(), Some("very"));
        assert_eq!(
            joiner.push("Very good", true, false).as_deref(),
            Some(". Very good")
        );
    }

    #[test]
    fn empty_transcript_breaks_the_pending_boundary() {
        let mut joiner = BoundaryTextJoiner::default();
        assert_eq!(
            joiner.push("first word", false, false).as_deref(),
            Some("first word")
        );
        assert_eq!(joiner.break_boundary(), None);
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
            Some("first word")
        );
        assert_eq!(joiner.break_boundary(), None);
        assert_eq!(
            joiner.push("Word again", true, true).as_deref(),
            Some("Word again")
        );
    }
}
