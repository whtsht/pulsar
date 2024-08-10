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

    pub fn peek_token(&mut self) -> Result<Token, ParseError> {
        self.lexer.peek_token().map_err(ParseError::LexerError)
    }

    pub fn skip_token(&mut self) {
        self.lexer.skip_token();
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

    pub fn parse_symbol(&mut self) -> Result<String, ParseError> {
        let token = self.next_token()?;
        token
            .as_symbol()
            .map(|s| s.to_string())
            .ok_or(ParseError::ExpectedSymbol(token))
    }

    pub fn parse_special_symbol(&mut self, sym: &str) -> Result<(), ParseError> {
        let token = self.next_token()?;
        if token.as_symbol() == Some(sym) {
            Ok(())
        } else {
            Err(ParseError::ExpectedSymbol(token))
        }
    }

    pub fn parse_lambda(&mut self) -> Result<Exp, ParseError> {
        self.skip_token();

        let token = self.next_token()?;
        let param = token
            .as_symbol()
            .ok_or(ParseError::ExpectedSymbol(token.clone()))?;
        let body = self.parse_exp()?;

        self.parse_right_param()?;

        Ok(lambda(param, body))
    }

    pub fn parse_exps(&mut self) -> Result<Vec<Exp>, ParseError> {
        let mut elems = Vec::new();
        while let Ok(token) = self.peek_token() {
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

        Ok(elems)
    }

    pub fn parse_if(&mut self) -> Result<Exp, ParseError> {
        self.skip_token();
        let cond = self.parse_exp()?;
        let then = self.parse_exp()?;
        let else_ = self.parse_exp()?;
        self.parse_right_param()?;
        Ok(if_(cond, then, else_))
    }

    pub fn parse_let(&mut self) -> Result<Exp, ParseError> {
        self.skip_token();

        self.parse_left_param()?;
        let token = self.next_token()?;
        let var = token
            .as_symbol()
            .ok_or(ParseError::ExpectedSymbol(token.clone()))?;
        let bind = (var, self.parse_exp()?);
        self.parse_right_param()?;

        let exp = self.parse_exp()?;

        self.parse_right_param()?;
        Ok(let_(bind, exp))
    }

    pub fn parse_one_case(&mut self) -> Result<(Exp, Exp), ParseError> {
        self.parse_left_param()?;
        let key = self.parse_exp()?;
        let value = self.parse_exp()?;
        self.parse_right_param()?;
        Ok((key, value))
    }

    pub fn parse_exp(&mut self) -> Result<Exp, ParseError> {
        let token = self.next_token()?;
        match token.kind {
            TokenKind::Quote => Ok(quote(self.parse_exp()?)),
            TokenKind::UnQuote => Ok(unquote(self.parse_exp()?)),
            TokenKind::Integer(int) => Ok(integer(int)),
            TokenKind::String(s) => Ok(Exp::String(s)),
            TokenKind::Symbol(sym) => match sym.as_str() {
                "nil" => Ok(nil()),
                "false" => Ok(bool(false)),
                "true" => Ok(bool(true)),
                sym => Ok(symbol(sym)),
            },
            TokenKind::LParen => match self.peek_token().map(|t| t.kind) {
                Ok(TokenKind::Symbol(sym)) => match sym.as_str() {
                    "\\" => self.parse_lambda(),
                    "if" => self.parse_if(),
                    "let" => self.parse_let(),
                    _ => Ok(list(&self.parse_exps()?)),
                },
                _ => Ok(list(&self.parse_exps()?)),
            },
            TokenKind::RParen => Err(ParseError::UnmatchedParen(token)),
        }
    }

    pub fn parse_def(&mut self) -> Result<(String, Exp), ParseError> {
        let name = self.parse_symbol()?;

        self.parse_left_param()?;
        let args = self.parse_exps()?;

        let body = self.parse_exp()?;

        self.parse_right_param()?;

        let exp = args
            .into_iter()
            .rev()
            .fold(body, |acc, arg| lambda(arg.as_symbol().unwrap(), acc));

        Ok((name, exp))
    }

    pub fn parse_macro(&mut self) -> Result<(String, Exp, Vec<String>), ParseError> {
        let name = self.parse_symbol()?;

        self.parse_left_param()?;
        let args = self
            .parse_exps()?
            .into_iter()
            .filter_map(|exp| exp.as_symbol().map(|sym| sym.to_string()))
            .collect();

        let body = self.parse_exp()?;

        self.parse_right_param()?;

        Ok((name, body, args))
    }

    pub fn parse_defines_or_macros(
        &mut self,
    ) -> Result<(Vec<(String, Exp)>, Vec<(String, Exp, Vec<String>)>), ParseError> {
        let mut defines = vec![];
        let mut macros = vec![];
        while let Ok(token) = self.peek_token() {
            if token.kind == TokenKind::RParen {
                break;
            }
            self.parse_left_param()?;

            match self.next_token()?.as_symbol() {
                Some("define") => defines.push(self.parse_def()?),
                Some("macro") => macros.push(self.parse_macro()?),
                _ => return Err(ParseError::ExpectedSymbol(token)),
            }
        }
        Ok((defines, macros))
    }

    pub fn parse_module(
        &mut self,
    ) -> Result<(String, Vec<(String, Exp)>, Vec<(String, Exp, Vec<String>)>), ParseError> {
        self.parse_left_param()?;
        self.parse_special_symbol("module")?;

        let module_name = self.parse_symbol()?;
        let (defines, macros) = self.parse_defines_or_macros()?;

        self.parse_right_param()?;

        Ok((module_name, defines, macros))
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

        let mut parser = Parser::new("'quit");
        assert_eq!(parser.parse_exp(), Ok(quote(symbol("quit"))));

        let mut parser = Parser::new("'(a b ~c)");
        assert_eq!(
            parser.parse_exp(),
            Ok(quote(list(&vec![
                symbol("a"),
                symbol("b"),
                unquote(symbol("c"))
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

    #[test]
    fn test_parse_module() {
        let mut parser = Parser::new("(module main (define foo (x) (+ 2 4)) (define bar () 2))");
        assert_eq!(
            parser.parse_module(),
            Ok((
                "main".to_string(),
                vec![
                    (
                        "foo".to_string(),
                        lambda("x", list(&vec![symbol("+"), integer(2), integer(4)]))
                    ),
                    ("bar".to_string(), integer(2))
                ],
                vec![]
            ))
        );
    }

    #[test]
    fn test_comment() {
        let mut parser = Parser::new(
            r#"(module main ; comment
            ; comment 
            (define foo (x) (+ 2 4)) ; comment
            ; comment
            (define bar () ; comment
             ; comment
             2))"#,
        );
        assert_eq!(
            parser.parse_module(),
            Ok((
                "main".to_string(),
                vec![
                    (
                        "foo".to_string(),
                        lambda("x", list(&vec![symbol("+"), integer(2), integer(4)]))
                    ),
                    ("bar".to_string(), integer(2))
                ],
                vec![]
            ))
        );
    }

    #[test]
    fn test_macro() {
        let mut parser = Parser::new(
            r#"
            (module test
              (macro unless (cond then else)
                '(if ~cond ~else ~then)))"#,
        );

        assert_eq!(
            parser.parse_module(),
            Ok((
                "test".to_string(),
                vec![],
                vec![(
                    "unless".to_string(),
                    quote(if_(
                        unquote(symbol("cond")),
                        unquote(symbol("else")),
                        unquote(symbol("then"))
                    )),
                    vec!["cond".into(), "then".into(), "else".into()],
                )]
            ))
        );
    }
}
