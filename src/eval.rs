use crate::{ast::*, buildin::default_module};

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum EvalError {
    IsNotNumber(Exp),
    InvalidArgs,
    DivideByZero,
    SymbolNotFound(String),
    Unexpected,
    ExpectedBool,
    ExpectedLambda,
    FailedToApply(Exp, Exp),
}

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
        Exp::Case(e, cases) => Exp::Case(
            Box::new(subst(e2.clone(), x.clone(), *e, gen)),
            cases
                .into_iter()
                .map(|(cond, body)| {
                    if let Exp::Symbol(sym) = &cond {
                        let var = gen.gen_var();
                        let body = subst(symbol(&var), sym.clone(), body, gen);
                        (symbol(&var), subst(e2.clone(), x.clone(), body, gen))
                    } else {
                        (
                            subst(e2.clone(), x.clone(), cond, gen),
                            subst(e2.clone(), x.clone(), body, gen),
                        )
                    }
                })
                .collect(),
        ),
        Exp::Quote(_) => e1,
        Exp::List(list) => Exp::List(
            list.into_iter()
                .map(|e| subst(e2.clone(), x.clone(), e, gen))
                .collect(),
        ),
    }
}

fn eval_app(
    e1: Exp,
    e2: Exp,
    module: &Module,
    gen: &mut VariableGenerator,
) -> Result<(Exp, bool), EvalError> {
    match e1 {
        Exp::Lambda(x, e11) => {
            if is_value(&e2) {
                let e = subst(e2, x, *e11, gen);
                Ok((e, true))
            } else {
                let (e2, _) = step(e2, module, gen)?;
                Ok((
                    Exp::Apply(Box::new(Exp::Lambda(x, e11)), Box::new(e2)),
                    true,
                ))
            }
        }
        Exp::Symbol(sym) => {
            if let Some(e1) = module.get(&sym) {
                eval_app(e1.clone(), e2, module, gen)
            } else {
                Err(EvalError::SymbolNotFound(sym))
            }
        }
        Exp::Apply(..) => {
            let (e1, _) = step(e1, module, gen)?;
            Ok((Exp::Apply(Box::new(e1), Box::new(e2)), true))
        }
        Exp::List(es) => {
            let (e1, _) = step(Exp::List(es), module, gen)?;
            Ok((Exp::Apply(Box::new(e1), Box::new(e2)), true))
        }
        _ => Err(EvalError::FailedToApply(e1, e2)),
    }
}

fn step(exp: Exp, module: &Module, gen: &mut VariableGenerator) -> Result<(Exp, bool), EvalError> {
    match exp {
        Exp::Integer(_) | Exp::Nil | Exp::Bool(_) | Exp::String(_) | Exp::BuildIn(_) => {
            Ok((exp, false))
        }
        Exp::Symbol(sym) => {
            if let Some(e) = module.get(&sym) {
                Ok((e.clone(), true))
            } else {
                Err(EvalError::SymbolNotFound(sym))
            }
        }
        Exp::Lambda(..) => Ok((exp, false)),
        Exp::Apply(e1, e2) => eval_app(*e1, *e2, module, gen),
        Exp::If(e1, e2, e3) => match eval(*e1, module, gen)? {
            Exp::Bool(true) => Ok((*e2, true)),
            Exp::Bool(false) => Ok((*e3, true)),
            _ => Err(EvalError::ExpectedBool),
        },
        Exp::Let(bind, exp) => {
            let (sym, body) = bind;
            Ok((apply(lambda(&sym, *exp), *body), true))
        }
        Exp::Case(e, cases) => {
            if is_value(&e) {
                for (cond, body) in cases {
                    if let Exp::Symbol(sym) = &cond {
                        return Ok((subst(*e, sym.clone(), body, gen), true));
                    }
                    if *e == cond {
                        return Ok((body, true));
                    }
                }
                todo!()
            } else {
                Ok((step(*e, module, gen)?.0, true))
            }
        }
        Exp::Quote(e) => Ok((*e, false)),
        Exp::List(list) => {
            if let Some((head, tail)) = list.split_first() {
                let head = eval(head.clone(), module, gen)?;

                if let Exp::BuildIn(f) = head {
                    let args = tail.iter().cloned().collect::<Vec<_>>();
                    Ok((f(&args, module, gen)?, false))
                } else {
                    let e = tail.iter().fold(head.clone(), |acc, e| {
                        Exp::Apply(Box::new(acc), Box::new(e.clone()))
                    });
                    Ok((e, true))
                }
            } else {
                Ok((Exp::Nil, false))
            }
        }
    }
}

pub fn eval_empty_module(mut exp: Exp) -> Result<Exp, EvalError> {
    let mut gen = VariableGenerator::new();
    loop {
        match step(exp, &Module::new("empty"), &mut gen) {
            Ok((exp_next, true)) => exp = exp_next,
            Ok((exp_next, false)) => return Ok(exp_next),
            Err(err) => return Err(err),
        }
    }
}

pub fn eval_default_module(mut exp: Exp) -> Result<Exp, EvalError> {
    let mut gen = VariableGenerator::new();
    loop {
        match step(exp, &default_module(), &mut gen) {
            Ok((exp_next, true)) => exp = exp_next,
            Ok((exp_next, false)) => return Ok(exp_next),
            Err(err) => return Err(err),
        }
    }
}

pub fn eval(mut exp: Exp, module: &Module, gen: &mut VariableGenerator) -> Result<Exp, EvalError> {
    loop {
        match step(exp, module, gen) {
            Ok((exp_next, true)) => exp = exp_next,
            Ok((exp_next, false)) => return Ok(exp_next),
            Err(err) => return Err(err),
        }
    }
}

impl Module {
    pub fn run(&self, name: &str, args: Vec<Exp>) -> Result<Exp, EvalError> {
        let mut exp = self
            .get(name)
            .map(|e| e.clone())
            .ok_or_else(|| EvalError::SymbolNotFound(name.to_string()))?;
        let mut gen = VariableGenerator::new();
        for arg in args {
            exp = apply(exp, arg);
        }
        eval(exp, self, &mut gen)
    }

    pub fn eval(&self, exp: Exp) -> Result<Exp, EvalError> {
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

        // [2/x](case x ((1 1) (2 2) (_ 0)))
        // => (case 2 ((1 1) (2 2) (#0 0)))
        let e = subst(
            integer(2),
            "x".to_string(),
            case(
                symbol("x"),
                &[
                    (integer(1), integer(1)),
                    (integer(2), integer(2)),
                    (symbol("_"), integer(0)),
                ],
            ),
            &mut VariableGenerator::new(),
        );
        assert_eq!(
            e,
            case(
                integer(2),
                &[
                    (integer(1), integer(1)),
                    (integer(2), integer(2)),
                    (symbol("#0"), integer(0)),
                ],
            )
        );

        // [2/x](case 1 (1 x) 0)
        // => (case 1 (1 2) 0)
        let e = subst(
            integer(2),
            "x".to_string(),
            case(integer(1), &[(integer(1), symbol("x"))]),
            &mut VariableGenerator::new(),
        );
        assert_eq!(e, case(integer(1), &[(integer(1), integer(2))]));

        // [2/x] (case 1 (x x))
        // => (case 1 (#0 #0))
        let e = subst(
            integer(2),
            "x".to_string(),
            case(integer(1), &[(symbol("x"), symbol("x"))]),
            &mut VariableGenerator::new(),
        );
        assert_eq!(e, case(integer(1), &[(symbol("#0"), symbol("#0"))],));

        // [2/x] (case 1 (y (+ y x)))
        // => (case 1 (#0 (+ #0 2)))
        let e = subst(
            integer(2),
            "x".to_string(),
            case(
                integer(1),
                &[(symbol("y"), list(&[symbol("+"), symbol("y"), symbol("x")]))],
            ),
            &mut VariableGenerator::new(),
        );
        assert_eq!(
            e,
            case(
                integer(1),
                &[(symbol("#0"), list(&[symbol("+"), symbol("#0"), integer(2)]))],
            )
        );
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
                    list(&[symbol("=="), symbol("n"), integer(0)]),
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
    fn test_case() {
        // (case 1
        //   (1 "one")
        //   (2 "two")
        //   (_ "other")
        // )
        // => "one"
        let e = case(
            integer(1),
            &[
                (integer(1), string("one")),
                (integer(2), string("two")),
                (symbol("_"), string("other")),
            ],
        );
        assert_eq!(eval_empty_module(e), Ok(string("one")));

        // (case nil
        //   (1 "one")
        //   (2 "two")
        //   (_ "other")
        // )
        // => "other"
        let e = case(
            integer(3),
            &[
                (integer(1), string("one")),
                (integer(2), string("two")),
                (symbol("_"), string("other")),
            ],
        );
        assert_eq!(eval_empty_module(e), Ok(string("other")));

        // (case 3
        //   (1 "one")
        //   (2 "two")
        //   (x (+ x 1))
        // )
        // => 4
        let e = case(
            integer(3),
            &[
                (integer(1), string("one")),
                (integer(2), string("two")),
                (symbol("x"), list(&[symbol("+"), symbol("x"), integer(1)])),
            ],
        );
        assert_eq!(eval_default_module(e), Ok(integer(4)));

        // (case 3
        //   (1 "one")
        //   (x x)
        //   (_ "other")
        // )
        let e = case(
            integer(3),
            &[
                (integer(1), string("one")),
                (symbol("x"), symbol("x")),
                (symbol("_"), string("other")),
            ],
        );
        assert_eq!(eval_empty_module(e), Ok(integer(3)));
    }

    #[test]
    fn test_load_func_with_arg() {
        let source = r#"
        (define test (a b c d) (+ (* a b) (* c d)))
        "#;
        let module = load_module("test", source).unwrap();
        println!("{:?}", module.get("test"));
        assert_eq!(module.defines.len(), default_module().defines.len() + 1);
        assert_eq!(
            module.run("test", vec![integer(2), integer(5), integer(1), integer(3)]),
            Ok(integer(13))
        );
    }
}
