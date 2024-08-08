use crate::{ast::*, buildin::default_module};

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum EvalError {
    IsNotNumber(Exp),
    InvalidArgs(Vec<Exp>),
    DivideByZero(Exp),
    SymbolNotFound(String),
    Unexpected(Exp),
    ExpectedBool(Exp),
    ExpectedLambda(Exp),
    FailedToApply(Exp, Exp),
    NeverMatched(Exp),
    UnquoteOutsideQuote(Exp),
}

pub type Result<T> = std::result::Result<T, EvalError>;

fn is_value(e: &Exp) -> bool {
    matches!(
        e,
        Exp::Integer(_)
            | Exp::Bool(_)
            | Exp::Nil
            | Exp::String(_)
            | Exp::Quote(_)
            | Exp::Symbol(_)
            | Exp::Lambda(..)
    )
}

pub struct VariableGenerator {
    counter: u64,
}

impl VariableGenerator {
    pub fn new() -> Self {
        Self { counter: 0 }
    }

    pub fn gen_var(&mut self) -> String {
        let var = format!("#{}", self.counter);
        self.counter += 1;
        var
    }
}

// [e2/x]e1
fn subst(e2: Exp, x: String, e1: Exp, gen: &mut VariableGenerator) -> Exp {
    match e1 {
        Exp::Nil | Exp::Integer(_) | Exp::Bool(_) | Exp::String(_) | Exp::BuildIn(_) => e1,
        Exp::Lambda(y, e) => {
            let yy = gen.gen_var();
            Exp::Lambda(
                yy.clone(),
                Box::new(subst(e2, x, subst(Exp::Symbol(yy), y, *e, gen), gen)),
            )
        }
        Exp::Apply(e11, e12) => Exp::Apply(
            Box::new(subst(e2.clone(), x.clone(), *e11, gen)),
            Box::new(subst(e2, x, *e12, gen)),
        ),
        Exp::Symbol(sym) => {
            if sym == x {
                e2
            } else {
                Exp::Symbol(sym)
            }
        }
        Exp::If(e11, e12, e13) => Exp::If(
            Box::new(subst(e2.clone(), x.clone(), *e11, gen)),
            Box::new(subst(e2.clone(), x.clone(), *e12, gen)),
            Box::new(subst(e2, x, *e13, gen)),
        ),
        Exp::Let(bind, e) => {
            let (sym, body) = bind;
            let e1 = apply(lambda(&sym, *e), *body);
            subst(e2, x, e1, gen)
        }
        Exp::Quote(e11) => Exp::Quote(Box::new(subst_unquote(e2, x, *e11, gen))),
        Exp::UnQuote(_) => unreachable!("subst unquote"),
        Exp::List(list) => Exp::List(
            list.into_iter()
                .map(|e| subst(e2.clone(), x.clone(), e, gen))
                .collect(),
        ),
    }
}

fn subst_unquote(e2: Exp, x: String, e1: Exp, gen: &mut VariableGenerator) -> Exp {
    match e1 {
        Exp::Nil
        | Exp::Bool(_)
        | Exp::Integer(_)
        | Exp::String(_)
        | Exp::Symbol(_)
        | Exp::BuildIn(_) => e1,
        Exp::List(es) => list(
            &es.into_iter()
                .map(|e| subst_unquote(e2.clone(), x.clone(), e, gen))
                .collect::<Vec<_>>(),
        ),
        Exp::Lambda(s, e) => lambda(&s, subst_unquote(e2, x, *e, gen)),
        Exp::Apply(e1, e2) => apply(
            subst_unquote(*e2.clone(), x.clone(), *e1, gen),
            subst_unquote(*e2.clone(), x, *e2, gen),
        ),
        Exp::If(c, t, e) => if_(
            subst_unquote(e2.clone(), x.clone(), *c, gen),
            subst_unquote(e2.clone(), x.clone(), *t, gen),
            subst_unquote(e2, x, *e, gen),
        ),
        Exp::Quote(e) => *e,
        Exp::UnQuote(e11) => unquote(subst(e2, x, *e11, gen)),
        Exp::Let((s, b), e) => let_(
            (&s, subst_unquote(e2.clone(), x.clone(), *b, gen)),
            subst_unquote(e2, x, *e, gen),
        ),
    }
}

fn eval_app(e1: Exp, e2: Exp, module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    match e1 {
        Exp::Lambda(x, e11) => {
            if is_value(&e2) {
                let e = subst(e2, x, *e11, gen);
                eval(e, module, gen)
            } else {
                let e2 = eval(e2, module, gen)?;
                eval(
                    Exp::Apply(Box::new(Exp::Lambda(x, e11)), Box::new(e2)),
                    module,
                    gen,
                )
            }
        }
        Exp::Symbol(sym) => {
            if let Some(e1) = module.defines.get(&sym) {
                eval_app(e1.clone(), e2, module, gen)
            } else {
                Err(EvalError::SymbolNotFound(sym))
            }
        }
        Exp::Apply(..) => {
            let e1 = eval(e1, module, gen)?;
            eval(Exp::Apply(Box::new(e1), Box::new(e2)), module, gen)
        }
        Exp::List(es) => {
            let e1 = eval(Exp::List(es), module, gen)?;
            eval(Exp::Apply(Box::new(e1), Box::new(e2)), module, gen)
        }
        Exp::BuildIn(f) => f(&[e2], module, gen),
        _ => Err(EvalError::FailedToApply(e1, e2)),
    }
}

pub fn eval(exp: Exp, module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    match exp.clone() {
        Exp::Integer(_) | Exp::Nil | Exp::Bool(_) | Exp::String(_) | Exp::BuildIn(_) => Ok(exp),
        Exp::Symbol(sym) => {
            if let Some(e) = module.defines.get(&sym) {
                Ok(e.clone())
            } else if let Some(_) = module.macros.get(&sym) {
                Ok(exp)
            } else {
                Err(EvalError::SymbolNotFound(sym))
            }
        }
        Exp::Lambda(..) => Ok(exp),
        Exp::Apply(e1, e2) => eval_app(*e1, *e2, module, gen),
        Exp::If(e1, e2, e3) => match eval(*e1, module, gen)? {
            Exp::Bool(true) => eval(*e2, module, gen),
            Exp::Bool(false) => eval(*e3, module, gen),
            _ => Err(EvalError::ExpectedBool(exp)),
        },
        Exp::Let(bind, exp) => {
            let (sym, body) = bind;
            eval(apply(lambda(&sym, *exp), *body), module, gen)
        }
        Exp::Quote(e) => eval_unquote(*e, module, gen),
        Exp::UnQuote(e) => Err(EvalError::UnquoteOutsideQuote(*e)),
        Exp::List(list) => {
            if let Some((head, tail)) = list.split_first() {
                let head = eval(head.clone(), module, gen)?;

                if let Some(sym) = head.as_symbol() {
                    if let Some((mut macro_, args)) = module.macros.get(sym).cloned() {
                        if args.len() == tail.len() {
                            for (arg, exp) in args.iter().zip(tail.iter()) {
                                if let Some(arg) = arg.as_symbol() {
                                    macro_ =
                                        subst(quote(exp.clone()), arg.to_string(), macro_, gen);
                                } else {
                                    return Err(EvalError::InvalidArgs(tail.to_vec()));
                                }
                            }
                            let expanded = eval(macro_, module, gen)?;
                            return eval(expanded, module, gen);
                        } else {
                            return Err(EvalError::InvalidArgs(tail.to_vec()));
                        }
                    }
                }

                if let Exp::BuildIn(f) = head {
                    let args = tail.to_vec();
                    f(&args, module, gen)
                } else {
                    let e = tail.iter().fold(head.clone(), |acc, e| {
                        Exp::Apply(Box::new(acc), Box::new(e.clone()))
                    });
                    eval(e, module, gen)
                }
            } else {
                Ok(Exp::Nil)
            }
        }
    }
}

fn eval_unquote(exp: Exp, module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    match exp {
        Exp::Nil
        | Exp::Bool(_)
        | Exp::Integer(_)
        | Exp::String(_)
        | Exp::Symbol(_)
        | Exp::BuildIn(_)
        | Exp::Quote(_) => Ok(exp),
        Exp::List(es) => Ok(Exp::List(
            es.into_iter()
                .map(|e| eval_unquote(e, module, gen))
                .collect::<Result<_>>()?,
        )),
        Exp::Lambda(s, e) => Ok(lambda(&s, eval_unquote(*e, module, gen)?)),
        Exp::Apply(e1, e2) => Ok(apply(
            eval_unquote(*e1, module, gen)?,
            eval_unquote(*e2, module, gen)?,
        )),
        Exp::If(c, t, e) => Ok(if_(
            eval_unquote(*c, module, gen)?,
            eval_unquote(*t, module, gen)?,
            eval_unquote(*e, module, gen)?,
        )),
        Exp::UnQuote(e) => eval(*e, module, gen),
        Exp::Let((s, b), e) => Ok(let_((&s, *b), eval_unquote(*e, module, gen)?)),
    }
}

pub fn eval_macro(macro_: Exp, module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    match macro_ {
        Exp::Lambda(param, body) => Ok(Exp::Lambda(param, body)),
        Exp::List(es) => {
            let mut es = es.into_iter().map(|e| eval_macro(e, module, gen));
            let head = es.next().ok_or(EvalError::NeverMatched(Exp::Nil))?;
            let head = head?;
            let tail = es.collect::<Result<Vec<_>>>()?;
            Ok(Exp::List(
                [head].iter().chain(tail.iter()).cloned().collect(),
            ))
        }
        _ => Ok(macro_),
    }
}

pub fn eval_empty_module(exp: Exp) -> Result<Exp> {
    let mut gen = VariableGenerator::new();
    let module = Module::new("empty");
    eval(exp, &module, &mut gen)
}

pub fn eval_default_module(exp: Exp) -> Result<Exp> {
    let mut gen = VariableGenerator::new();
    let module = default_module();
    eval(exp, &module, &mut gen)
}

impl Module {
    pub fn run(&self, name: &str, args: Vec<Exp>) -> Result<Exp> {
        let mut exp = self
            .defines
            .get(name)
            .cloned()
            .ok_or_else(|| EvalError::SymbolNotFound(name.to_string()))?;
        let mut gen = VariableGenerator::new();
        for arg in args {
            exp = apply(exp, arg);
        }
        eval(exp, self, &mut gen)
    }

    pub fn eval(&self, exp: Exp) -> Result<Exp> {
        let mut gen = VariableGenerator::new();
        eval(exp, self, &mut gen)
    }
}

#[cfg(test)]
mod test {
    use crate::{buildin::default_module, loader::load_module};

    use super::*;

    #[test]
    fn test_subst() {
        // [2/x]x => 2
        let e = subst(
            integer(2),
            "x".to_string(),
            symbol("x"),
            &mut VariableGenerator::new(),
        );
        assert_eq!(e, integer(2));

        // [2/x]((\ x 1) x)
        // => ((\ #0 1) 2)
        let e = subst(
            integer(2),
            "x".to_string(),
            apply(lambda("x", integer(1)), symbol("x")),
            &mut VariableGenerator::new(),
        );
        assert_eq!(e, apply(lambda("#0", integer(1)), integer(2)));

        // [2/x]((\ x x) 1)
        // => ((\ #0 #0) 1)
        let e = subst(
            integer(2),
            "x".to_string(),
            apply(lambda("x", symbol("x")), integer(1)),
            &mut VariableGenerator::new(),
        );
        assert_eq!(e, apply(lambda("#0", symbol("#0")), integer(1)));

        // [2/x](let (x 1) x)
        // => [2/x]((\ x x) 1)
        // => ((\ #0 #0) 1)
        let e = subst(
            integer(2),
            "x".to_string(),
            let_(("x", integer(1)), symbol("x")),
            &mut VariableGenerator::new(),
        );
        assert_eq!(e, apply(lambda("#0", symbol("#0")), integer(1)));

        // [2/x](let (y 1) x)
        // => [2/x]((\y x) 1)
        // => ((\#0 2) 1)
        let e = subst(
            integer(2),
            "x".to_string(),
            let_(("y", integer(1)), symbol("x")),
            &mut VariableGenerator::new(),
        );
        assert_eq!(e, apply(lambda("#0", integer(2)), integer(1)));
    }

    #[test]
    fn test_atom() {
        // Nil => Nil
        let e = nil();
        assert_eq!(eval_empty_module(e), Ok(nil()));

        // 1 => 1
        let e = integer(1);
        assert_eq!(eval_empty_module(e), Ok(integer(1)));

        // "hello" => "hello"
        let e = string("hello");
        assert_eq!(eval_empty_module(e), Ok(string("hello")));

        // (quote ((\ x x) 1)) => ((\ x x) 1)
        let e = quote(apply(lambda("x", symbol("x")), integer(1)));
        assert_eq!(
            eval_empty_module(e),
            Ok(apply(lambda("x", symbol("x")), integer(1)))
        );

        // (quote a) => a
        let e = quote(symbol("a"));
        assert_eq!(eval_empty_module(e), Ok(symbol("a")));
    }

    #[test]
    fn test_lambda() {
        // (((\x . (\y . y)) 'a) 'b) => b
        let e = apply(
            apply(lambda("x", lambda("y", symbol("y"))), quote(symbol("a"))),
            quote(symbol("b")),
        );
        assert_eq!(eval_empty_module(e), Ok(symbol("b")));

        // // ((\x . (\y . y)) 'a 'b) => b
        let e = apply(
            apply(lambda("x", lambda("y", symbol("y"))), quote(symbol("a"))),
            quote(symbol("b")),
        );
        assert_eq!(eval_empty_module(e), Ok(symbol("b")));

        // ((\x (\y (x y))) y)
        // => (\#0 (y #0))
        let e = apply(
            lambda("x", lambda("y", apply(symbol("x"), symbol("y")))),
            symbol("y"),
        );
        assert_eq!(
            eval_empty_module(e),
            Ok(lambda("#0", apply(symbol("y"), symbol("#0")))),
        );
    }

    #[test]
    fn test_if() {
        // (if true 1 2) => 1
        let e = if_(bool(true), integer(1), integer(2));
        assert_eq!(eval_empty_module(e), Ok(integer(1)));

        // (if false 1 2) => 2
        let e = if_(bool(false), integer(1), integer(2));
        assert_eq!(eval_empty_module(e), Ok(integer(2)));
    }

    #[test]
    fn test_frac() {
        let mut module = default_module();
        module.defines.insert(
            "frac".to_string(),
            lambda(
                "n",
                if_(
                    list(&[symbol("="), symbol("n"), integer(0)]),
                    integer(1),
                    list(&[
                        symbol("*"),
                        symbol("n"),
                        apply(
                            symbol("frac"),
                            list(&[symbol("-"), symbol("n"), integer(1)]),
                        ),
                    ]),
                ),
            ),
        );
        let exp = list(&[symbol("frac"), integer(5)]);
        let mut gen = VariableGenerator::new();
        assert_eq!(eval(exp, &module, &mut gen), Ok(integer(120)));
    }

    #[test]
    fn test_let() {
        // (let (a 1) (+ a 10))
        // => ((\ a (+ a 10)) 1)
        // => 11
        let e = let_(
            ("a", integer(1)),
            list(&[symbol("+"), symbol("a"), integer(10)]),
        );
        assert_eq!(eval_default_module(e), Ok(integer(11)));
        // (let (a 1) (let (b a) (+ a b)))
        // => (let (b 1) (+ 1 b))
        // => (+ 1 1)
        // => 2
        let e = let_(
            ("a", integer(1)),
            let_(
                ("b", symbol("a")),
                list(&[symbol("+"), symbol("a"), symbol("b")]),
            ),
        );
        assert_eq!(eval_default_module(e), Ok(integer(2)));
    }

    #[test]
    fn test_run_func_with_arg() {
        let source = r#"
        (module test
            (define test (a b c d) (+ (* a b) (* c d))))
        "#;
        let module = load_module(source).unwrap();
        assert_eq!(module.defines.len(), default_module().defines.len() + 1);
        assert_eq!(
            module.run("test", vec![integer(2), integer(5), integer(1), integer(3)]),
            Ok(integer(13))
        );
    }

    #[test]
    fn test_unquote() {
        let source = r#"
        (module test
            (define test () '(a ~(+ 1 2))))
        "#;
        let module = load_module(source).unwrap();
        assert_eq!(module.defines.len(), default_module().defines.len() + 1);
        assert_eq!(
            module.run("test", vec![]),
            Ok(list(&[symbol("a"), integer(3)]))
        );

        let source = r#"
        (module test
            (define test (a) '(a ~a)))
        "#;
        let module = load_module(source).unwrap();
        assert_eq!(module.defines.len(), default_module().defines.len() + 1);
        assert_eq!(
            module.run("test", vec![integer(1)]),
            Ok(list(&[symbol("a"), integer(1)]))
        );

        // '(a '(b ~(+ 1 2)))
        // => (a '(b ~(+ 1 2)))
        let e = quote(list(&[
            symbol("a"),
            quote(list(&[
                symbol("b"),
                unquote(list(&[symbol("+"), integer(1), integer(2)])),
            ])),
        ]));
        assert_eq!(
            eval_empty_module(e),
            Ok(list(&[
                symbol("a"),
                quote(list(&[
                    symbol("b"),
                    unquote(list(&[symbol("+"), integer(1), integer(2)]))
                ]))
            ]))
        );
    }

    #[test]
    fn test_macro() {
        let source = r#"
        (module test
            (macro unless (cond then else) '(if ~cond ~else ~then))
            (macro and (a b) '(if ~a ~b ~a))
            (macro or (a b) '(if ~a ~a ~b))
            (define test1 () (unless (= 1 1) (/ 1 0) 'b))
            (define test2 () (and false (/ 1 0)))
            (define test3 () (or true (/ 1 0))))"#;
        let module = load_module(source).unwrap();
        assert_eq!(module.defines.len(), default_module().defines.len() + 3);
        assert_eq!(module.run("test1", vec![]), Ok(symbol("b")));
        assert_eq!(module.run("test2", vec![]), Ok(bool(false)));
        assert_eq!(module.run("test3", vec![]), Ok(bool(true)));
    }
}
