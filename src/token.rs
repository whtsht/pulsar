#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Location {
    pub line: usize,
    pub column: usize,
}

impl Location {
    pub fn new(line: usize, column: usize) -> Location {
        Location { line, column }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub loc: Location,
}

impl Token {
    pub fn new(kind: TokenKind, loc: Location) -> Self {
        Token { kind, loc }
    }
}

impl Token {
    pub fn as_symbol(&self) -> Option<&str> {
        match &self.kind {
            TokenKind::Symbol(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum TokenKind {
    LParen,
    RParen,
    Integer(i64),
    Symbol(String),
    String(String),
    Quote,
    BackQuote,
    UnQuote,
    Extend,
    Spread,
}

fn separator(ch: char) -> bool {
    matches!(ch, '\n') || ch.is_whitespace()
}

pub fn get_token_word(start: Location, input: &str) -> String {
    let s = input.split('\n').nth(start.line).unwrap();
    s.chars()
        .skip(start.column)
        .take_while(|&ch| !separator(ch))
        .collect()
}
