//! Post-processing applied to every transcription, all backends.
//!
//! Curly quotes literally break wtype, and a stray newline presses Enter in the
//! target app — in a terminal that *executes* the line. Both get neutralized.

use text2num::{replace_numbers_in_text, Language};

/// Trim, normalize curly quotes to ASCII, collapse newlines to spaces, then —
/// if `itn_numbers` — rewrite spoken numbers as digits ("twenty five" → "25").
pub fn post_process(s: &str, itn_numbers: bool) -> String {
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
    if itn_numbers {
        // threshold 2.0: isolated single-digit cardinals/ordinals below it stay
        // as words, so "no one" / "at one point" survive; "two"+ and every
        // multi-word number ("twenty five", "ninety nine") still convert.
        replace_numbers_in_text(out, &Language::english(), 2.0)
    } else {
        out.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::post_process;

    fn pp(s: &str) -> String {
        post_process(s, true)
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
    fn itn_converts_spoken_numbers() {
        assert_eq!(
            pp("set timer for twenty five minutes"),
            "set timer for 25 minutes"
        );
        assert_eq!(pp("version two point one"), "version 2.1");
        assert_eq!(pp("ninety nine percent"), "99 percent");
        assert_eq!(pp("i need two things"), "i need 2 things");
    }

    #[test]
    fn itn_guards_isolated_ones() {
        // The classic ITN false positives — "one" in prose must stay a word.
        assert_eq!(pp("no one came"), "no one came");
        assert_eq!(pp("at one point i thought"), "at one point i thought");
        assert_eq!(pp("one of the best"), "one of the best");
    }

    #[test]
    fn itn_off_leaves_words() {
        assert_eq!(
            post_process("twenty five minutes", false),
            "twenty five minutes"
        );
    }
}
