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
    ExpectedIdent(Token),
}

pub type Result<T> = std::result::Result<T, ParseError>;

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
        ParseError::ExpectedIdent(token) => {
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

    pub fn next_token(&mut self) -> Result<Token> {
        self.lexer.next_token().map_err(ParseError::LexerError)
    }

    pub fn peek_token(&mut self) -> Result<Token> {
        self.lexer.peek_token().map_err(ParseError::LexerError)
    }

    pub fn skip_token(&mut self) {
        self.lexer.skip_token();
    }

    pub fn parse_right_param(&mut self) -> Result<()> {
        match self.next_token()? {
            token if token.kind == TokenKind::RParen => Ok(()),
            token => Err(ParseError::UnmatchedParen(token)),
        }
    }

    pub fn parse_left_param(&mut self) -> Result<()> {
        match self.next_token()? {
            token if token.kind == TokenKind::LParen => Ok(()),
            token => Err(ParseError::UnmatchedParen(token)),
        }
    }

    pub fn parse_chars(&mut self) -> Result<String> {
        let token = self.next_token()?;
        token
            .as_ident()
            .map(|s| s.to_string())
            .ok_or(ParseError::ExpectedIdent(token))
    }

    pub fn parse_special_chars(&mut self, sym: &str) -> Result<()> {
        let token = self.next_token()?;
        if token.as_ident() == Some(sym) {
            Ok(())
        } else {
            Err(ParseError::ExpectedIdent(token))
        }
    }

    pub fn parse_symbol(&mut self, first: &str) -> Result<Symbol> {
        let mut symbols = vec![first.to_string()];
        while let Ok(TokenKind::NameResolver) = self.peek_token().map(|token| token.kind) {
            self.skip_token();
            symbols.push(self.parse_chars()?);
        }

        let (name, symbols) = symbols.split_last().unwrap();
        Ok(Symbol {
            name: name.to_string(),
            namespace: symbols.to_vec(),
        })
    }

    pub fn parse_lambda(&mut self) -> Result<Exp> {
        self.skip_token();

        let token = self.next_token()?;
        let param = token
            .as_ident()
            .ok_or(ParseError::ExpectedIdent(token.clone()))?;
        let body = self.parse_exp()?;

        self.parse_right_param()?;

        Ok(lambda(param, body))
    }

    pub fn parse_exps(&mut self) -> Result<Vec<Exp>> {
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

    pub fn parse_if(&mut self) -> Result<Exp> {
        self.skip_token();
        let cond = self.parse_exp()?;
        let then = self.parse_exp()?;
        let else_ = self.parse_exp()?;
        self.parse_right_param()?;
        Ok(if_(cond, then, else_))
    }

    pub fn parse_let(&mut self) -> Result<Exp> {
        self.skip_token();

        self.parse_left_param()?;
        let token = self.next_token()?;
        let var = token
            .as_ident()
            .ok_or(ParseError::ExpectedIdent(token.clone()))?;
        let bind = (var, self.parse_exp()?);
        self.parse_right_param()?;

        let exp = self.parse_exp()?;

        self.parse_right_param()?;
        Ok(let_(bind, exp))
    }

    pub fn parse_one_case(&mut self) -> Result<(Exp, Exp)> {
        self.parse_left_param()?;
        let key = self.parse_exp()?;
        let value = self.parse_exp()?;
        self.parse_right_param()?;
        Ok((key, value))
    }

    pub fn parse_exp(&mut self) -> Result<Exp> {
        let token = self.next_token()?;
        match token.kind {
            TokenKind::Quote => Ok(quote(self.parse_exp()?)),
            TokenKind::BackQuote => Ok(backquote(self.parse_exp()?)),
            TokenKind::UnQuote => Ok(unquote(self.parse_exp()?)),
            TokenKind::Extend => Ok(extend(self.parse_exp()?)),
            TokenKind::Integer(int) => Ok(integer(int)),
            TokenKind::String(s) => Ok(Exp::String(s)),
            TokenKind::Import => todo!(),
            TokenKind::Ident(sym) => match sym.as_str() {
                "nil" => Ok(nil()),
                "false" => Ok(bool(false)),
                "true" => Ok(bool(true)),
                sym => Ok(Exp::Symbol(self.parse_symbol(sym)?)),
            },
            TokenKind::LParen => match self.peek_token().map(|t| t.kind) {
                Ok(TokenKind::Ident(sym)) => match sym.as_str() {
                    "\\" => self.parse_lambda(),
                    "if" => self.parse_if(),
                    "let" => self.parse_let(),
                    _ => Ok(list(&self.parse_exps()?)),
                },
                _ => Ok(list(&self.parse_exps()?)),
            },
            TokenKind::RParen => Err(ParseError::UnmatchedParen(token)),
            TokenKind::Spread => Err(ParseError::ExpectedIdent(token)),
            TokenKind::NameResolver => unreachable!(),
        }
    }

    pub fn parse_args(&mut self) -> Result<Vec<String>> {
        let mut elems = Vec::new();
        while let Ok(token) = self.peek_token() {
            match token.kind {
                TokenKind::RParen => {
                    break;
                }
                _ => {
                    elems.push(self.parse_chars()?);
                }
            }
        }
        self.parse_right_param()?;

        Ok(elems)
    }

    pub fn parse_def(&mut self) -> Result<Define> {
        let name = self.parse_chars()?;

        self.parse_left_param()?;
        let args = self.parse_args()?;

        let body = self
            .parse_exps()?
            .into_iter()
            .map(|exp| {
                args.clone()
                    .into_iter()
                    .rev()
                    .fold(exp, |acc, arg| lambda(&arg, acc))
            })
            .collect::<Vec<_>>();

        Ok(Define {
            name,
            exp: list(&[vec![symbol("block", vec![])], body].concat()),
        })
    }

    pub fn parse_macro_args(&mut self) -> Result<(Vec<String>, Option<String>)> {
        let mut args = Vec::new();
        while let Ok(token) = self.peek_token() {
            match token.kind {
                TokenKind::RParen => {
                    break;
                }
                TokenKind::Spread => {
                    self.skip_token();
                    let var_arg = Some(self.parse_chars()?);
                    self.parse_right_param()?;
                    return Ok((args, var_arg));
                }
                _ => {
                    args.push(self.parse_chars()?);
                }
            }
        }
        self.parse_right_param()?;

        Ok((args, None))
    }

    pub fn parse_macro(&mut self) -> Result<Macro> {
        let name = self.parse_chars()?;

        self.parse_left_param()?;
        let (args, var_arg) = self.parse_macro_args()?;

        let exp = self.parse_exp()?;

        self.parse_right_param()?;

        Ok(Macro {
            name,
            exp,
            args,
            var_arg,
        })
    }

    pub fn parse_import(&mut self) -> Result<String> {
        let ident = self.parse_chars()?;
        self.parse_right_param()?;
        Ok(ident)
    }

    pub fn parse_inner_module(&mut self) -> Result<UnresolvedModule> {
        let name = self.parse_chars()?;
        let module = self.parse_module(&name)?;
        self.parse_right_param()?;
        Ok(module)
    }

    pub fn parse_defines_or_macros(&mut self) -> Result<(Option<Define>, Option<Macro>)> {
        self.parse_left_param()?;
        let token = self.next_token()?;
        match token.as_ident() {
            Some("define") => Ok((Some(self.parse_def()?), None)),
            Some("macro") => Ok((None, Some(self.parse_macro()?))),
            _ => return Err(ParseError::ExpectedIdent(token)),
        }
    }

    pub fn parse_module(&mut self, name: &str) -> Result<UnresolvedModule> {
        let mut defines = vec![];
        let mut macros = vec![];
        let mut inner_modules = vec![];
        let mut imported = vec![];

        while let Ok(token) = self.peek_token() {
            if token.kind == TokenKind::RParen {
                break;
            }
            self.parse_left_param()?;

            match self.next_token()?.as_ident() {
                Some("define") => defines.push(self.parse_def()?),
                Some("macro") => macros.push(self.parse_macro()?),
                Some("module") => inner_modules.push(self.parse_inner_module()?),
                Some("import") => imported.push(self.parse_import()?),
                _ => return Err(ParseError::ExpectedIdent(token)),
            }
        }

        Ok(UnresolvedModule {
            name: name.to_string(),
            defines,
            macros,
            inner_modules,
            imported,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::Parser;
    use crate::ast::*;

    fn func(exp: Exp) -> Exp {
        list(&vec![symbol("block", vec![]), exp])
    }

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
        assert_eq!(parser.parse_exp(), Ok(symbol("hello", vec![])));

        let mut parser = Parser::new("_");
        assert_eq!(parser.parse_exp(), Ok(symbol("_", vec![])));
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
                symbol("html", vec![]),
                list(&vec![
                    symbol("head", vec![]),
                    quote(list(&vec![
                        symbol("body", vec![]),
                        list(&vec![symbol("h1", vec![]), string("hello")])
                    ]))
                ])
            ])))
        );

        let mut parser = Parser::new("'quit");
        assert_eq!(parser.parse_exp(), Ok(quote(symbol("quit", vec![]))));

        let mut parser = Parser::new("'(a b .c)");
        assert_eq!(
            parser.parse_exp(),
            Ok(quote(list(&vec![
                symbol("a", vec![]),
                symbol("b", vec![]),
                unquote(symbol("c", vec![]))
            ])))
        );
    }

    #[test]
    fn test_parse_lambda() {
        let mut parser = Parser::new(r"(\ x x)");
        assert_eq!(parser.parse_exp(), Ok(lambda("x", symbol("x", vec![]))));
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
                    list(&vec![
                        symbol("+", vec![]),
                        symbol("x", vec![]),
                        symbol("y", vec![])
                    ])
                )
            ))
        );
    }

    #[test]
    fn test_parse_list() {
        let mut parser = Parser::new("(+ 1 2)");
        assert_eq!(
            parser.parse_exp(),
            Ok(list(&vec![symbol("+", vec![]), integer(1), integer(2)]))
        );

        let mut parser = Parser::new("(+ 1 (+ 2 3))");
        assert_eq!(
            parser.parse_exp(),
            Ok(list(&vec![
                symbol("+", vec![]),
                integer(1),
                list(&vec![symbol("+", vec![]), integer(2), integer(3)])
            ]))
        );

        let mut parser = Parser::new("()");
        assert_eq!(parser.parse_exp(), Ok(list(&vec![])));
    }

    #[test]
    fn test_comment() {
        let mut parser = Parser::new(
            r#"
            ; comment 
            (define foo (x) (+ 2 4)) ; comment
            ; comment
            (define bar () ; comment
             ; comment
             2)"#,
        );
        assert_eq!(
            parser.parse_module("test"),
            Ok(UnresolvedModule {
                name: "test".to_string(),
                defines: vec![
                    Define::new(
                        "foo",
                        func(lambda(
                            "x",
                            list(&vec![symbol("+", vec![]), integer(2), integer(4)])
                        ))
                    ),
                    Define::new("bar", func(integer(2)))
                ],
                ..Default::default()
            })
        );
    }

    #[test]
    fn test_macro() {
        let mut parser = Parser::new(
            r#"
            (macro unless (cond then else)
              `(if .cond .else .then))"#,
        );

        assert_eq!(
            parser.parse_module("test"),
            Ok(UnresolvedModule {
                name: "test".to_string(),
                macros: vec![Macro::new(
                    "unless",
                    backquote(if_(
                        unquote(symbol("cond", vec![])),
                        unquote(symbol("else", vec![])),
                        unquote(symbol("then", vec![]))
                    )),
                    vec!["cond".into(), "then".into(), "else".into()],
                    None
                )],
                ..Default::default()
            })
        );

        let mut parser = Parser::new(
            r#"
            (macro sum (...values)
              `(foldl + 0 '.values))"#,
        );
        assert_eq!(
            parser.parse_module("test"),
            Ok(UnresolvedModule {
                name: "test".to_string(),
                macros: vec![Macro::new(
                    "sum",
                    backquote(list(&vec![
                        symbol("foldl", vec![]),
                        symbol("+", vec![]),
                        integer(0),
                        quote(unquote(symbol("values", vec![])))
                    ])),
                    vec![],
                    Some("values".into())
                )],
                ..Default::default()
            })
        );
    }

    #[test]
    fn test_import() {
        let mut parser = Parser::new(
            r#"
            (import foo)
            (import bar)
            (define test () nil)
            "#,
        );
        assert_eq!(
            parser.parse_module("test"),
            Ok(UnresolvedModule {
                name: "test".to_string(),
                imported: vec!["foo".into(), "bar".into()],
                defines: vec![Define::new("test", func(nil()))],
                ..Default::default()
            })
        );
    }

    #[test]
    fn test_symbol() {
        let mut parser = Parser::new("foo");
        assert_eq!(
            parser.parse_symbol("foo"),
            Ok(Symbol::new("foo".to_string(), vec![]))
        );

        let mut parser = Parser::new("::bar::baz");
        assert_eq!(
            parser.parse_symbol("foo"),
            Ok(Symbol::new(
                "baz".to_string(),
                vec!["foo".to_string(), "bar".to_string()]
            ))
        );
    }

    #[test]
    fn test_call_module() {
        let mut parser = Parser::new(
            r#"
            (module foo
              (define x () 1))
            (define test () foo::x)
            "#,
        );
        assert_eq!(
            parser.parse_module("test"),
            Ok(UnresolvedModule {
                name: "test".to_string(),
                defines: vec![Define::new(
                    "test",
                    func(symbol("x", vec!["foo".to_string()]))
                )],
                inner_modules: vec![UnresolvedModule {
                    name: "foo".to_string(),
                    defines: vec![Define::new("x", func(integer(1)))],
                    ..Default::default()
                }],
                ..Default::default()
            })
        );
    }
}
