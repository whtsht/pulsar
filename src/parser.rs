use crate::ast::*;
use crate::lexer::{lexer_error_message, Lexer, LexerError};
use crate::token::{get_token_word, Token, TokenKind};

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Parser {
    lexer: Lexer,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ParseError {
    LexerError(LexerError),
    UnmatchedParen(Token),
    ExpectedSymbol(Token),
}

pub fn parse_error_message(error: ParseError, input: &str) -> String {
    match error {
        ParseError::LexerError(err) => lexer_error_message(err, input),
        ParseError::UnmatchedParen(token) => {
            let word = get_token_word(token.loc, input);
            format!(
                "{}\n{} unexpected parentheses",
                word,
                "^".repeat(word.len())
            )
        }
        ParseError::ExpectedSymbol(token) => {
            let word = get_token_word(token.loc, input);
            format!("{}\n{} expected symbol", word, "^".repeat(word.len()))
        }
    }
}

impl Parser {
    pub fn new(input: &str) -> Parser {
        Parser {
            lexer: Lexer::new(input),
        }
    }

    pub fn next_token(&mut self) -> Result<Token, ParseError> {
        self.lexer.next_token().map_err(ParseError::LexerError)
    }

    pub fn parse_right_param(&mut self) -> Result<(), ParseError> {
        match self.next_token()? {
            token if token.kind == TokenKind::RParen => Ok(()),
            token => Err(ParseError::UnmatchedParen(token)),
        }
    }

    pub fn parse_left_param(&mut self) -> Result<(), ParseError> {
        match self.next_token()? {
            token if token.kind == TokenKind::LParen => Ok(()),
            token => Err(ParseError::UnmatchedParen(token)),
        }
    }

    pub fn parse_lambda(&mut self) -> Result<Exp, ParseError> {
        self.lexer.skip_token();

        let token = self.next_token()?;
        let param = token
            .as_symbol()
            .ok_or(ParseError::ExpectedSymbol(token.clone()))?;
        let body = self.parse_exp()?;

        self.parse_right_param()?;

        Ok(lambda(param, body))
    }

    pub fn parse_list(&mut self) -> Result<Exp, ParseError> {
        let mut elems = Vec::new();
        while let Ok(token) = self.lexer.peek_token() {
            match token.kind {
                TokenKind::RParen => {
                    break;
                }
                _ => {
                    elems.push(self.parse_exp()?);
                }
            }
        }
        self.parse_right_param()?;

        Ok(list(&elems))
    }

    pub fn parse_if(&mut self) -> Result<Exp, ParseError> {
        self.lexer.skip_token();
        Ok(if_(self.parse_exp()?, self.parse_exp()?, self.parse_exp()?))
    }

    pub fn parse_let(&mut self) -> Result<Exp, ParseError> {
        self.lexer.skip_token();

        self.parse_left_param()?;
        let token = self.next_token()?;
        let var = token
            .as_symbol()
            .ok_or(ParseError::ExpectedSymbol(token.clone()))?;
        let bind = (var, self.parse_exp()?);
        self.parse_right_param()?;

        let exp = self.parse_exp()?;

        Ok(let_(bind, exp))
    }

    pub fn parse_one_case(&mut self) -> Result<(Exp, Exp), ParseError> {
        self.parse_left_param()?;
        let key = self.parse_exp()?;
        let value = self.parse_exp()?;
        self.parse_right_param()?;
        Ok((key, value))
    }

    pub fn parse_cases(&mut self) -> Result<Exp, ParseError> {
        self.lexer.skip_token();

        let exp = self.parse_exp()?;
        let mut cases = Vec::new();
        while let Ok(TokenKind::LParen) = self.lexer.peek_token().map(|t| t.kind) {
            cases.push(self.parse_one_case()?);
        }

        Ok(case(exp, &cases))
    }

    pub fn parse_exp(&mut self) -> Result<Exp, ParseError> {
        let token = self.lexer.next_token().map_err(ParseError::LexerError)?;
        match token.kind {
            TokenKind::Integer(int) => Ok(integer(int)),
            TokenKind::String(s) => Ok(Exp::String(s)),
            TokenKind::Symbol(sym) => match sym.as_str() {
                "nil" => Ok(nil()),
                "false" => Ok(bool(false)),
                "true" => Ok(bool(true)),
                "'" => Ok(quote(self.parse_exp()?)),
                sym => Ok(symbol(sym)),
            },
            TokenKind::LParen => match self.lexer.peek_token().map(|t| t.kind) {
                Ok(TokenKind::Symbol(sym)) => match sym.as_str() {
                    "\\" => self.parse_lambda(),
                    "if" => self.parse_if(),
                    "let" => self.parse_let(),
                    "case" => self.parse_cases(),
                    _ => self.parse_list(),
                },
                _ => self.parse_list(),
            },
            TokenKind::RParen => Err(ParseError::UnmatchedParen(token)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Parser;
    use crate::ast::*;

    #[test]
    fn test_parse_integer() {
        let mut parser = Parser::new("123");
        assert_eq!(parser.parse_exp(), Ok(integer(123)));
    }

    #[test]
    fn test_parse_nil() {
        let mut parser = Parser::new("nil");
        assert_eq!(parser.parse_exp(), Ok(nil()));
    }

    #[test]
    fn test_parse_bool() {
        let mut parser = Parser::new("true");
        assert_eq!(parser.parse_exp(), Ok(bool(true)));

        let mut parser = Parser::new("false");
        assert_eq!(parser.parse_exp(), Ok(bool(false)));
    }

    #[test]
    fn test_parse_symbol() {
        let mut parser = Parser::new("hello");
        assert_eq!(parser.parse_exp(), Ok(symbol("hello")));

        let mut parser = Parser::new("_");
        assert_eq!(parser.parse_exp(), Ok(symbol("_")));
    }

    #[test]
    fn test_parse_string() {
        let mut parser = Parser::new(r#""hello""#);
        assert_eq!(parser.parse_exp(), Ok(string("hello")));
    }

    #[test]
    fn test_parse_quote() {
        let mut parser = Parser::new("'(html (head '(body (h1 \"hello\"))))");
        assert_eq!(
            parser.parse_exp(),
            Ok(quote(list(&vec![
                symbol("html"),
                list(&vec![
                    symbol("head"),
                    quote(list(&vec![
                        symbol("body"),
                        list(&vec![symbol("h1"), string("hello")])
                    ]))
                ])
            ])))
        );
    }

    #[test]
    fn test_parse_lambda() {
        let mut parser = Parser::new(r"(\ x x)");
        assert_eq!(parser.parse_exp(), Ok(lambda("x", symbol("x"))));
    }

    #[test]
    fn test_parse_if() {
        let mut parser = Parser::new("(if true 1 2)");
        assert_eq!(
            parser.parse_exp(),
            Ok(if_(bool(true), integer(1), integer(2)))
        );
    }

    #[test]
    fn test_parse_let() {
        let mut parser = Parser::new("(let (x 1) 2)");
        assert_eq!(parser.parse_exp(), Ok(let_(("x", integer(1)), integer(2))));

        let mut parser = Parser::new("(let (x 1) (let (y 2) (+ x y)))");
        assert_eq!(
            parser.parse_exp(),
            Ok(let_(
                ("x", integer(1)),
                let_(
                    ("y", integer(2)),
                    list(&vec![symbol("+"), symbol("x"), symbol("y")])
                )
            ))
        );
    }

    #[test]
    fn test_parse_case() {
        let mut parser = Parser::new("(case 1 (1 2) (2 3) (_ 0))");
        assert_eq!(
            parser.parse_exp(),
            Ok(case(
                integer(1),
                &vec![
                    (integer(1), integer(2)),
                    (integer(2), integer(3)),
                    (symbol("_"), integer(0))
                ]
            ))
        );
    }

    #[test]
    fn test_parse_list() {
        let mut parser = Parser::new("(+ 1 2)");
        assert_eq!(
            parser.parse_exp(),
            Ok(list(&vec![symbol("+"), integer(1), integer(2)]))
        );

        let mut parser = Parser::new("(+ 1 (+ 2 3))");
        assert_eq!(
            parser.parse_exp(),
            Ok(list(&vec![
                symbol("+"),
                integer(1),
                list(&vec![symbol("+"), integer(2), integer(3)])
            ]))
        );

        let mut parser = Parser::new("()");
        assert_eq!(parser.parse_exp(), Ok(list(&vec![])));
    }
}
