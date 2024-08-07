use crate::token::{get_token_word, Location, Token, TokenKind};
use std::ops::Not;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Lexer {
    input: Vec<char>,
    pos: usize,
    loc: Location,
    token: Option<Token>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum LexerError {
    IsNotInteger(Location),
    Eot(Location),
    InvalidSymbol(Location),
}

pub fn lexer_error_message(error: LexerError, input: &str) -> String {
    match error {
        LexerError::IsNotInteger(loc) => {
            let word = get_token_word(loc, input);
            format!("{}\n{} not a integer", word, "^".repeat(word.len()))
        }
        LexerError::Eot(_) => {
            let word = if input.len() < 10 {
                input
            } else {
                &input[input.len() - 10..]
            };
            format!(
                "{} \n{}^ unexpected end of token",
                word,
                " ".repeat(word.len())
            )
        }
        LexerError::InvalidSymbol(loc) => {
            let word = get_token_word(loc, input);
            format!("{}\n{} invalid symbol", word, "^".repeat(word.len()))
        }
    }
}

fn valid_sym(ch: char) -> bool {
    matches!(
       ch,
       'a'..='z' | 'A'..='Z' | '0'..='9'
       | '+' | '-' | '*' | '/'
       | '%' | '<' | '=' | '>'
       | '&' | '_' | '!' | '$'
       | '?' | '^' | '~' | '\\'
       | '@' | '\''| '"'
    )
}

fn separator(ch: char) -> bool {
    matches!(ch, '(' | ')' | '\n') || ch.is_whitespace()
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        Lexer {
            input: input.chars().collect(),
            pos: 0,
            loc: Location::new(0, 0),
            token: None,
        }
    }

    fn check_eof(&self) -> Result<(), LexerError> {
        if self.pos >= self.input.len() {
            Err(LexerError::Eot(self.loc))
        } else {
            Ok(())
        }
    }

    pub fn current_char(&self) -> Result<char, LexerError> {
        self.check_eof()?;
        Ok(self.input[self.pos])
    }

    pub fn inc(&mut self) -> Result<(), LexerError> {
        if self.current_char()? == '\n' {
            self.loc.line += 1;
            self.loc.column = 0
        } else {
            self.loc.column += 1;
        }
        self.pos += 1;
        Ok(())
    }

    pub fn dec(&mut self) {
        if self.pos == 0 {
            return;
        }
        self.pos -= 1;
        if self.input[self.pos] == '\n' {
            self.loc.line -= 1;
            self.loc.column = self.input[..self.pos]
                .iter()
                .rev()
                .take_while(|&&c| c != '\n')
                .count();
        } else {
            self.loc.column -= 1;
        }
    }

    pub fn next_cher(&mut self) -> Result<char, LexerError> {
        self.check_eof()?;
        let ch = self.input[self.pos];
        self.inc()?;
        Ok(ch)
    }

    pub fn symbol(&mut self) -> Result<String, LexerError> {
        let loc = self.loc;
        let ch = self.next_cher()?;
        if separator(ch) {
            self.dec();
            return Ok(String::new());
        }
        if ch.is_ascii_digit() || valid_sym(ch).not() {
            return Err(LexerError::InvalidSymbol(loc));
        }
        let mut value = String::from(ch);

        while let Ok(ch) = self.next_cher() {
            if separator(ch) {
                self.dec();
                break;
            }
            if !valid_sym(ch) {
                return Err(LexerError::InvalidSymbol(loc));
            }
            value.push(ch);
        }

        Ok(value)
    }

    pub fn numbers(&mut self) -> Result<String, LexerError> {
        let loc = self.loc;
        let mut value = String::new();
        while let Ok(ch) = self.next_cher() {
            if separator(ch) {
                self.dec();
                break;
            }
            if !ch.is_ascii_digit() {
                return Err(LexerError::IsNotInteger(loc));
            }
            value.push(ch);
        }
        Ok(value)
    }

    pub fn peek_token(&mut self) -> Result<Token, LexerError> {
        if let Some(token) = &self.token {
            Ok(token.clone())
        } else {
            let token = self.next_token()?;
            self.token = Some(token.clone());
            Ok(token)
        }
    }

    pub fn next_token(&mut self) -> Result<Token, LexerError> {
        if let Some(token) = self.token.take() {
            return Ok(token);
        }

        while self.current_char()?.is_whitespace() {
            self.inc()?;
        }

        match self.input[self.pos] {
            '\'' => {
                let loc = self.loc;
                self.inc()?;
                Ok(Token::new(TokenKind::Quote, loc))
            }
            '~' => {
                let loc = self.loc;
                self.inc()?;
                Ok(Token::new(TokenKind::UnQuote, loc))
            }
            '(' => {
                let loc = self.loc;
                self.inc()?;
                Ok(Token::new(TokenKind::LParen, loc))
            }
            ')' => {
                let loc = self.loc;
                self.inc()?;
                Ok(Token::new(TokenKind::RParen, loc))
            }
            '0' => {
                let loc = self.loc;
                let value = self.numbers()?;
                Ok(Token::new(
                    TokenKind::Integer(value.parse().map_err(|_| LexerError::IsNotInteger(loc))?),
                    loc,
                ))
            }
            '1'..='9' => {
                let loc = self.loc;
                let value = self.numbers()?;
                Ok(Token::new(
                    TokenKind::Integer(value.parse().map_err(|_| LexerError::IsNotInteger(loc))?),
                    loc,
                ))
            }
            '-' => {
                let loc = self.loc;
                self.inc()?;
                if let Some('0'..='9') = self.input.get(self.pos) {
                    let value = format!("-{}", self.numbers()?);
                    Ok(Token::new(
                        TokenKind::Integer(
                            value.parse().map_err(|_| LexerError::IsNotInteger(loc))?,
                        ),
                        loc,
                    ))
                } else {
                    let value = format!("-{}", self.symbol()?);
                    Ok(Token::new(TokenKind::Symbol(value), loc))
                }
            }
            '"' => {
                let mut value = String::new();
                let loc = self.loc;
                self.inc()?;
                while self.input[self.pos] != '"' {
                    value.push(self.next_cher()?);
                }
                self.inc()?;
                Ok(Token::new(TokenKind::String(value), loc))
            }
            _ => {
                let loc = self.loc;
                let value = self.symbol()?;
                Ok(Token::new(TokenKind::Symbol(value), loc))
            }
        }
    }

    pub fn skip_token(&mut self) {
        self.next_token().ok();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_symbol() {
        let mut lexer = Lexer::new("hello world");
        assert_eq!(
            lexer.next_token(),
            Ok(Token::new(
                TokenKind::Symbol("hello".to_string()),
                Location::new(0, 0)
            ))
        );
        assert_eq!(
            lexer.next_token(),
            Ok(Token::new(
                TokenKind::Symbol("world".to_string()),
                Location::new(0, 6)
            ))
        );

        let mut lexer = Lexer::new("0aaa");
        assert_eq!(
            lexer.next_token(),
            Err(LexerError::IsNotInteger(Location::new(0, 0)))
        );

        let mut lexer = Lexer::new("a0aa");
        assert_eq!(
            lexer.next_token(),
            Ok(Token::new(
                TokenKind::Symbol("a0aa".to_string()),
                Location::new(0, 0)
            ))
        );

        let mut lexer = Lexer::new("<=has-many=>");
        assert_eq!(
            lexer.next_token(),
            Ok(Token::new(
                TokenKind::Symbol("<=has-many=>".to_string()),
                Location::new(0, 0)
            ))
        );

        let mut lexer = Lexer::new("aaa)bbb");
        assert_eq!(
            lexer.next_token(),
            Ok(Token::new(
                TokenKind::Symbol("aaa".to_string()),
                Location::new(0, 0)
            ))
        );
        assert_eq!(
            lexer.next_token(),
            Ok(Token::new(TokenKind::RParen, Location::new(0, 3)))
        );
        assert_eq!(
            lexer.next_token(),
            Ok(Token::new(
                TokenKind::Symbol("bbb".to_string()),
                Location::new(0, 4)
            ))
        );
    }

    #[test]
    fn test_string() {
        let mut lexer = Lexer::new(r#""hello world""#);
        assert_eq!(
            lexer.next_token(),
            Ok(Token::new(
                TokenKind::String("hello world".to_string()),
                Location::new(0, 0)
            ))
        );
    }

    #[test]
    fn test_number() {
        let mut lexer = Lexer::new("123 456");
        assert_eq!(
            lexer.next_token(),
            Ok(Token::new(TokenKind::Integer(123), Location::new(0, 0)))
        );
        assert_eq!(
            lexer.next_token(),
            Ok(Token::new(TokenKind::Integer(456), Location::new(0, 4)))
        );
        let mut lexer = Lexer::new("1a0");
        assert_eq!(
            lexer.next_token(),
            Err(LexerError::IsNotInteger(Location::new(0, 0)))
        );
    }

    #[test]
    fn test_list() {
        let mut lexer = Lexer::new("(+ 1 200)");
        assert_eq!(
            lexer.next_token(),
            Ok(Token::new(TokenKind::LParen, Location::new(0, 0)))
        );
        assert_eq!(
            lexer.next_token(),
            Ok(Token::new(
                TokenKind::Symbol("+".to_string()),
                Location::new(0, 1)
            ))
        );
        assert_eq!(
            lexer.next_token(),
            Ok(Token::new(TokenKind::Integer(1), Location::new(0, 3)))
        );
        assert_eq!(
            lexer.next_token(),
            Ok(Token::new(TokenKind::Integer(200), Location::new(0, 5)))
        );
        assert_eq!(
            lexer.next_token(),
            Ok(Token::new(TokenKind::RParen, Location::new(0, 8)))
        );
    }

    #[test]
    fn test_uniop_minus() {
        let mut lexer = Lexer::new("(- 1)");
        assert_eq!(
            lexer.next_token(),
            Ok(Token::new(TokenKind::LParen, Location::new(0, 0)))
        );
        assert_eq!(
            lexer.next_token(),
            Ok(Token::new(
                TokenKind::Symbol("-".to_string()),
                Location::new(0, 1)
            ))
        );
        assert_eq!(
            lexer.next_token(),
            Ok(Token::new(TokenKind::Integer(1), Location::new(0, 3)))
        );
        assert_eq!(
            lexer.next_token(),
            Ok(Token::new(TokenKind::RParen, Location::new(0, 4)))
        );

        let mut lexer = Lexer::new("(-1)");
        assert_eq!(
            lexer.next_token(),
            Ok(Token::new(TokenKind::LParen, Location::new(0, 0)))
        );
        assert_eq!(
            lexer.next_token(),
            Ok(Token::new(TokenKind::Integer(-1), Location::new(0, 1)))
        );
        assert_eq!(
            lexer.next_token(),
            Ok(Token::new(TokenKind::RParen, Location::new(0, 3)))
        );
    }

    #[test]
    fn test_loc() {
        let mut lexer = Lexer::new(
            r#"(hello
               world)"#,
        );

        assert_eq!(
            lexer.next_token(),
            Ok(Token::new(TokenKind::LParen, Location::new(0, 0)))
        );
        assert_eq!(
            lexer.next_token(),
            Ok(Token::new(
                TokenKind::Symbol("hello".to_string()),
                Location::new(0, 1)
            ))
        );
        assert_eq!(
            lexer.next_token(),
            Ok(Token::new(
                TokenKind::Symbol("world".to_string()),
                Location::new(1, 15)
            ))
        );
        assert_eq!(
            lexer.next_token(),
            Ok(Token::new(TokenKind::RParen, Location::new(1, 20)))
        );
    }
}
