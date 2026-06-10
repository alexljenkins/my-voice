//! Post-processing applied to every transcription, all backends.
//!
//! Curly quotes literally break wtype, and a stray newline presses Enter in the
//! target app — in a terminal that *executes* the line. Both get neutralized.

/// Trim, normalize curly quotes to ASCII, collapse newlines to spaces.
pub fn post_process(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        out.push(match c {
            '\u{2018}' | '\u{2019}' | '\u{201B}' | '\u{2032}' => '\'',
            '\u{201C}' | '\u{201D}' | '\u{201F}' | '\u{2033}' => '"',
            '\n' | '\r' => ' ',
            other => other,
        });
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::post_process;

    #[test]
    fn trims_whitespace() {
        assert_eq!(post_process("  hello  "), "hello");
    }

    #[test]
    fn normalizes_curly_quotes() {
        assert_eq!(post_process("\u{2018}hi\u{2019}"), "'hi'");
        assert_eq!(post_process("\u{201C}hi\u{201D}"), "\"hi\"");
        assert_eq!(post_process("it\u{2019}s"), "it's");
    }

    #[test]
    fn collapses_newlines() {
        assert_eq!(post_process("a\nb"), "a b");
        assert_eq!(post_process("ls\n"), "ls");
    }
}
