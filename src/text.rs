//! Post-processing applied to every transcription, all backends.
//!
//! Curly quotes literally break wtype, and a stray newline presses Enter in the
//! target app — in a terminal that *executes* the line. Both get neutralized.
//!
//! Optional inverse text normalization rewrites spoken numbers as digits. Two
//! independent toggles: `itn_numbers` for cardinals ("twenty five" → "25") and
//! `itn_ordinals` for ordinals ("second" → "2nd"). They are separate because
//! ordinal words double as everyday nouns ("give me a second"), so the default
//! is cardinals on, ordinals off.

use text2num::{find_numbers, Language, Token};

/// Isolated cardinals/ordinals below this value stay as words, so "no one" and
/// "at one point" survive while "two"+ and any multi-word number still convert.
const ITN_THRESHOLD: f64 = 2.0;

/// Trim, normalize curly quotes to ASCII, collapse newlines to spaces, then —
/// per the ITN toggles — rewrite spoken numbers and/or ordinals as digits.
pub fn post_process(s: &str, itn_numbers: bool, itn_ordinals: bool) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        out.push(match c {
            '\u{2018}' | '\u{2019}' | '\u{201B}' | '\u{2032}' => '\'',
            '\u{201C}' | '\u{201D}' | '\u{201F}' | '\u{2033}' => '"',
            '\n' | '\r' => ' ',
            other => other,
        });
    }
    let out = out.trim();
    if itn_numbers || itn_ordinals {
        itn(out, itn_numbers, itn_ordinals)
    } else {
        out.to_string()
    }
}

/// One token: a word (alphanumerics plus `-`/`'`) or a run of separators,
/// mirroring text2num's own tokenizer so concatenating `text` rebuilds the
/// input exactly — only the number spans we keep get swapped for digits.
#[derive(Clone)]
struct Tok {
    text: String,
    lower: String,
}

impl Token for Tok {
    fn text(&self) -> &str {
        &self.text
    }
    fn text_lowercase(&self) -> &str {
        &self.lower
    }
}

/// Split on word boundaries: an alphanumeric run (extended with `-`/`'`) is a
/// word, everything else is a separator run. Every byte is preserved.
fn tokenize(s: &str) -> Vec<Tok> {
    let mut toks = Vec::new();
    let mut chars = s.char_indices().peekable();
    while let Some((start, c)) = chars.next() {
        let word = c.is_alphanumeric();
        let mut end = s.len();
        while let Some(&(pos, c2)) = chars.peek() {
            let cont = if word {
                c2.is_alphanumeric() || c2 == '-' || c2 == '\''
            } else {
                !c2.is_alphanumeric()
            };
            if !cont {
                end = pos;
                break;
            }
            chars.next();
        }
        let text = s[start..end].to_string();
        let lower = text.to_lowercase();
        toks.push(Tok { text, lower });
    }
    toks
}

/// Replace number spans with their digit form, keeping cardinals iff `numbers`
/// and ordinals iff `ordinals`. text2num finds the spans and tags each with
/// `is_ordinal`; we splice in `occ.text` over the matched tokens and pass the
/// rest through — a filtered-out number stays as its original words.
fn itn(s: &str, numbers: bool, ordinals: bool) -> String {
    let toks = tokenize(s);
    let lang = Language::english();
    let mut keep = find_numbers(toks.iter().cloned(), &lang, ITN_THRESHOLD)
        .into_iter()
        .filter(|o| if o.is_ordinal { ordinals } else { numbers })
        .peekable();

    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < toks.len() {
        match keep.peek() {
            Some(o) if o.start == i => {
                out.push_str(&o.text);
                i = o.end;
                keep.next();
            }
            _ => {
                out.push_str(&toks[i].text);
                i += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::post_process;

    /// Default daemon ITN: cardinals on, ordinals off.
    fn pp(s: &str) -> String {
        post_process(s, true, false)
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
    fn cardinals_convert() {
        assert_eq!(
            pp("set timer for twenty five minutes"),
            "set timer for 25 minutes"
        );
        assert_eq!(pp("version two point one"), "version 2.1");
        assert_eq!(pp("ninety nine percent"), "99 percent");
        assert_eq!(pp("i need two things"), "i need 2 things");
    }

    #[test]
    fn isolated_low_numbers_stay_words() {
        // The classic ITN false positives — "one" in prose must stay a word.
        assert_eq!(pp("no one came"), "no one came");
        assert_eq!(pp("at one point i thought"), "at one point i thought");
        assert_eq!(pp("one of the best"), "one of the best");
    }

    #[test]
    fn ordinals_stay_words_by_default() {
        // The reason ordinals are a separate, default-off toggle: these words
        // are everyday nouns/adjectives far more often than ranks.
        assert_eq!(pp("give me a second"), "give me a second");
        assert_eq!(pp("the first thing"), "the first thing");
        assert_eq!(pp("on second thought"), "on second thought");
    }

    #[test]
    fn ordinals_convert_only_when_enabled() {
        // Cardinals + ordinals both on: a clearly-ranked "second" converts.
        assert_eq!(
            post_process("meeting on the second", true, true),
            "meeting on the 2nd"
        );
        // Cardinals stay numeric regardless of the ordinal toggle.
        assert_eq!(post_process("twenty five", true, true), "25");
        // Ordinals on, cardinals off: the cardinal stays a word, the ordinal converts.
        assert_eq!(
            post_process("two by the second", false, true),
            "two by the 2nd"
        );
    }

    #[test]
    fn both_off_is_plain_passthrough() {
        assert_eq!(
            post_process("twenty five minutes", false, false),
            "twenty five minutes"
        );
        assert_eq!(
            post_process("give me a second", false, false),
            "give me a second"
        );
    }

    #[test]
    fn punctuation_and_spacing_preserved() {
        assert_eq!(pp("i have two, maybe three"), "i have 2, maybe 3");
        assert_eq!(pp("hello world"), "hello world");
    }
}
