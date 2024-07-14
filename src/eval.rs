use crate::ast::*;
use std::collections::HashMap;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum EvalError {
    IsNotNumber(Exp),
    SymbolNotFound(String),
    Unexpected,
    ExpectedBool,
    ExpectedLambda,
    FailedToApply,
}

pub struct Env {
    defines: HashMap<String, Exp>,
    counter: u64,
}

impl Env {
    pub fn new() -> Self {
        Self {
            defines: HashMap::new(),
            counter: 0,
        }
    }

    pub fn insert(&mut self, key: String, value: Exp) {
        self.defines.insert(key, value);
    }

    pub fn get(&self, key: &String) -> Option<&Exp> {
        self.defines.get(key)
    }

    pub fn gen_var(&mut self) -> String {
        let var = format!("#{}", self.counter);
        self.counter += 1;
        var
    }
}

fn parse_list_of_integer(args: &[Exp]) -> Result<Vec<i64>, EvalError> {
    args.iter()
        .map(|x| match x {
            Exp::Integer(i) => Ok(*i),
            _ => Err(EvalError::IsNotNumber(x.clone())),
        })
        .collect()
}

fn add(args: &[Exp]) -> Result<Exp, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::Unexpected);
    }
    let args = parse_list_of_integer(args)?;
    Ok(Exp::Integer(args[0] + args[1]))
}

fn sub(args: &[Exp]) -> Result<Exp, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::Unexpected);
    }
    let args = parse_list_of_integer(args)?;
    Ok(Exp::Integer(args[0] - args[1]))
}

fn mul(args: &[Exp]) -> Result<Exp, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::Unexpected);
    }
    let args = parse_list_of_integer(args)?;
    Ok(Exp::Integer(args[0] * args[1]))
}

fn eq(args: &[Exp]) -> Result<Exp, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::Unexpected);
    }
    Ok(Exp::Bool(args[0] == args[1]))
}

fn default_env() -> Env {
    let mut env = Env::new();
    env.insert(
        "+".to_string(),
        Exp::Lambda(
            "x".to_string(),
            Box::new(Exp::Lambda(
                "y".to_string(),
                Box::new(Exp::BuildIn(
                    add,
                    vec![Exp::Symbol("x".to_string()), Exp::Symbol("y".to_string())],
                )),
            )),
        ),
    );
    env.insert(
        "-".to_string(),
        Exp::Lambda(
            "x".to_string(),
            Box::new(Exp::Lambda(
                "y".to_string(),
                Box::new(Exp::BuildIn(
                    sub,
                    vec![Exp::Symbol("x".to_string()), Exp::Symbol("y".to_string())],
                )),
            )),
        ),
    );
    env.insert(
        "*".to_string(),
        Exp::Lambda(
            "x".to_string(),
            Box::new(Exp::Lambda(
                "y".to_string(),
                Box::new(Exp::BuildIn(
                    mul,
                    vec![Exp::Symbol("x".to_string()), Exp::Symbol("y".to_string())],
                )),
            )),
        ),
    );
    env.insert(
        "eq?".to_string(),
        Exp::Lambda(
            "x".to_string(),
            Box::new(Exp::Lambda(
                "y".to_string(),
                Box::new(Exp::BuildIn(
                    eq,
                    vec![Exp::Symbol("x".to_string()), Exp::Symbol("y".to_string())],
                )),
            )),
        ),
    );

    env
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

// [e2/x]e1
fn subst(e2: Exp, x: String, e1: Exp, env: &mut Env) -> Exp {
    match e1 {
        Exp::Nil | Exp::Integer(_) | Exp::Bool(_) | Exp::String(_) => e1,
        Exp::Lambda(y, e) => {
            let yy = env.gen_var();
            Exp::Lambda(
                yy.clone(),
                Box::new(subst(e2, x, subst(Exp::Symbol(yy), y, *e, env), env)),
            )
        }
        Exp::Apply(e11, e12) => Exp::Apply(
            Box::new(subst(e2.clone(), x.clone(), *e11, env)),
            Box::new(subst(e2, x, *e12, env)),
        ),
        Exp::Symbol(sym) => {
            if sym == x {
                e2
            } else {
                Exp::Symbol(sym)
            }
        }
        Exp::BuildIn(func, args) => Exp::BuildIn(
            func,
            args.into_iter()
                .map(|e| subst(e2.clone(), x.clone(), e, env))
                .collect(),
        ),
        Exp::If(e11, e12, e13) => Exp::If(
            Box::new(subst(e2.clone(), x.clone(), *e11, env)),
            Box::new(subst(e2.clone(), x.clone(), *e12, env)),
            Box::new(subst(e2, x, *e13, env)),
        ),
        Exp::Let(bind, e) => {
            let (sym, body) = bind;
            let e1 = apply(lambda(&sym, *e), *body);
            subst(e2, x, e1, env)
        }
        Exp::Case(e, cases) => Exp::Case(
            Box::new(subst(e2.clone(), x.clone(), *e, env)),
            cases
                .into_iter()
                .map(|(cond, body)| {
                    if let Exp::Symbol(sym) = &cond {
                        let var = env.gen_var();
                        let body = subst(symbol(&var), sym.clone(), body, env);
                        (symbol(&var), subst(e2.clone(), x.clone(), body, env))
                    } else {
                        (
                            subst(e2.clone(), x.clone(), cond, env),
                            subst(e2.clone(), x.clone(), body, env),
                        )
                    }
                })
                .collect(),
        ),
        Exp::Quote(_) => e1,
        Exp::List(list) => Exp::List(
            list.into_iter()
                .map(|e| subst(e2.clone(), x.clone(), e, env))
                .collect(),
        ),
    }
}

fn eval_app(e1: Exp, e2: Exp, env: &mut Env) -> Result<(Exp, bool), EvalError> {
    match e1 {
        Exp::Lambda(x, e11) => {
            if is_value(&e2) {
                let e = subst(e2, x, *e11, env);
                Ok((e, true))
            } else {
                let (e2, _) = step(e2, env)?;
                Ok((
                    Exp::Apply(Box::new(Exp::Lambda(x, e11)), Box::new(e2)),
                    true,
                ))
            }
        }
        Exp::Symbol(sym) => {
            if let Some(e1) = env.get(&sym) {
                eval_app(e1.clone(), e2, env)
            } else {
                Err(EvalError::SymbolNotFound(sym))
            }
        }
        Exp::Apply(..) => {
            let (e1, _) = step(e1, env)?;
            Ok((Exp::Apply(Box::new(e1), Box::new(e2)), true))
        }
        Exp::List(es) => {
            let (e1, _) = step(Exp::List(es), env)?;
            Ok((Exp::Apply(Box::new(e1), Box::new(e2)), true))
        }
        _ => Err(EvalError::FailedToApply),
    }
}

fn step(e: Exp, env: &mut Env) -> Result<(Exp, bool), EvalError> {
    match e {
        Exp::Integer(_) | Exp::Nil | Exp::Bool(_) | Exp::String(_) => Ok((e, false)),
        Exp::Symbol(sym) => {
            if let Some(e) = env.get(&sym) {
                Ok((e.clone(), true))
            } else {
                Err(EvalError::SymbolNotFound(sym))
            }
        }
        Exp::Lambda(..) => Ok((e, false)),
        Exp::Apply(e1, e2) => eval_app(*e1, *e2, env),
        Exp::BuildIn(func, mut args) => {
            let mut arg = None;
            for (i, e) in args.iter().enumerate() {
                if !is_value(e) {
                    arg = Some((e.clone(), i));
                    break;
                }
            }
            if let Some((e, i)) = arg {
                let (e, _) = step(e, env)?;
                args[i] = e;
                Ok((Exp::BuildIn(func, args), true))
            } else {
                Ok((func(&args)?, false))
            }
        }
        Exp::If(e1, e2, e3) => match eval(*e1)? {
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
                        return Ok((subst(*e, sym.clone(), body, env), true));
                    }
                    if *e == cond {
                        return Ok((body, true));
                    }
                }
                todo!()
            } else {
                Ok((step(*e, env)?.0, true))
            }
        }
        Exp::Quote(e) => Ok((*e, false)),
        Exp::List(list) => {
            if let Some((head, tail)) = list.split_first() {
                let e = tail.iter().fold(head.clone(), |acc, e| {
                    Exp::Apply(Box::new(acc), Box::new(e.clone()))
                });
                Ok((e, true))
            } else {
                Ok((Exp::Nil, false))
            }
        }
    }
}

pub fn eval(mut e: Exp) -> Result<Exp, EvalError> {
    let mut env = default_env();
    loop {
        match step(e, &mut env) {
            Ok((e1, true)) => e = e1,
            Ok((e1, false)) => return Ok(e1),
            Err(e) => return Err(e),
        }
    }
}

pub fn eval_with_env(mut e: Exp, env: &mut Env) -> Result<Exp, EvalError> {
    loop {
        match step(e, env) {
            Ok((e1, true)) => e = e1,
            Ok((e1, false)) => return Ok(e1),
            Err(e) => return Err(e),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_subst() {
        // [2/x]x => 2
        let e = subst(integer(2), "x".to_string(), symbol("x"), &mut Env::new());
        assert_eq!(e, integer(2));

        // [2/x]((\ x 1) x)
        // => ((\ #0 1) 2)
        let e = subst(
            integer(2),
            "x".to_string(),
            apply(lambda("x", integer(1)), symbol("x")),
            &mut Env::new(),
        );
        assert_eq!(e, apply(lambda("#0", integer(1)), integer(2)));

        // [2/x]((\ x x) 1)
        // => ((\ #0 #0) 1)
        let e = subst(
            integer(2),
            "x".to_string(),
            apply(lambda("x", symbol("x")), integer(1)),
            &mut Env::new(),
        );
        assert_eq!(e, apply(lambda("#0", symbol("#0")), integer(1)));

        // [2/x](let (x 1) x)
        // => [2/x]((\ x x) 1)
        // => ((\ #0 #0) 1)
        let e = subst(
            integer(2),
            "x".to_string(),
            let_(("x", integer(1)), symbol("x")),
            &mut Env::new(),
        );
        assert_eq!(e, apply(lambda("#0", symbol("#0")), integer(1)));

        // [2/x](let (y 1) x)
        // => [2/x]((\y x) 1)
        // => ((\#0 2) 1)
        let e = subst(
            integer(2),
            "x".to_string(),
            let_(("y", integer(1)), symbol("x")),
            &mut Env::new(),
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
            &mut Env::new(),
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
            &mut Env::new(),
        );
        assert_eq!(e, case(integer(1), &[(integer(1), integer(2))]));

        // [2/x] (case 1 (x x))
        // => (case 1 (#0 #0))
        let e = subst(
            integer(2),
            "x".to_string(),
            case(integer(1), &[(symbol("x"), symbol("x"))]),
            &mut Env::new(),
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
            &mut Env::new(),
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
        assert_eq!(eval(e), Ok(nil()));

        // 1 => 1
        let e = integer(1);
        assert_eq!(eval(e), Ok(integer(1)));

        // "hello" => "hello"
        let e = string("hello");
        assert_eq!(eval(e), Ok(string("hello")));

        // (quote ((\ x x) 1)) => ((\ x x) 1)
        let e = quote(apply(lambda("x", symbol("x")), integer(1)));
        assert_eq!(eval(e), Ok(apply(lambda("x", symbol("x")), integer(1))));

        // (quote a) => a
        let e = quote(symbol("a"));
        assert_eq!(eval(e), Ok(symbol("a")));
    }

    #[test]
    fn test_lambda() {
        // (((\x . (\y . y)) 'a) 'b) => b
        let e = apply(
            apply(lambda("x", lambda("y", symbol("y"))), quote(symbol("a"))),
            quote(symbol("b")),
        );
        assert_eq!(eval(e), Ok(symbol("b")));

        // // ((\x . (\y . y)) 'a 'b) => b
        let e = apply(
            apply(lambda("x", lambda("y", symbol("y"))), quote(symbol("a"))),
            quote(symbol("b")),
        );
        assert_eq!(eval(e), Ok(symbol("b")));

        // ((\x (\y (x y))) y)
        // => (\#0 (y #0))
        let e = apply(
            lambda("x", lambda("y", apply(symbol("x"), symbol("y")))),
            symbol("y"),
        );
        assert_eq!(eval(e), Ok(lambda("#0", apply(symbol("y"), symbol("#0")))),);
    }

    #[test]
    fn test_eq() {
        let eq = |a, b| list(&[symbol("eq?"), a, b]);
        // (eq? 1 1) => true
        let e = eq(integer(1), integer(1));
        assert_eq!(eval(e), Ok(bool(true)));
    }

    #[test]
    fn test_if() {
        // (if true 1 2) => 1
        let e = if_(bool(true), integer(1), integer(2));
        assert_eq!(eval(e), Ok(integer(1)));

        // (if false 1 2) => 2
        let e = if_(bool(false), integer(1), integer(2));
        assert_eq!(eval(e), Ok(integer(2)));
    }

    #[test]
    fn test_add() {
        // ((\x . (\y . (+ x y))) 1 2)
        let e = list(&[
            lambda(
                "x",
                lambda("y", list(&[symbol("+"), symbol("x"), symbol("y")])),
            ),
            integer(1),
            integer(2),
        ]);
        assert_eq!(eval(e), Ok(Exp::Integer(3)));

        // (+ 1)
        // => \#0 (BUILDIN(+) 1 #0)
        let e = list(&[symbol("+"), integer(1)]);
        assert_eq!(
            eval(e),
            Ok(lambda("#0", buildin(add, &[integer(1), symbol("#0")])))
        );
    }

    #[test]
    fn test_frac() {
        let mut env = default_env();
        env.insert(
            "frac".to_string(),
            lambda(
                "n",
                if_(
                    list(&[symbol("eq?"), symbol("n"), integer(0)]),
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
        let e = list(&[symbol("frac"), integer(5)]);
        assert_eq!(eval_with_env(e, &mut env), Ok(integer(120)));
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
        assert_eq!(eval(e), Ok(integer(11)));
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
        assert_eq!(eval(e), Ok(integer(2)));
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
        assert_eq!(eval(e), Ok(string("one")));

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
        assert_eq!(eval(e), Ok(string("other")));

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
        assert_eq!(eval(e), Ok(integer(4)));

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
        assert_eq!(eval(e), Ok(integer(3)));
    }

    #[test]
    fn test_call_by_value() {
        fn step_by(e: Exp, cnt: u32, env: &mut Env) -> Exp {
            let mut e = e;
            for _ in 0..cnt {
                e = step(e, env).unwrap().0;
            }
            e
        }
        // ((\ x (\ y (y x))) (+ 2 2) (\ x (+ x 1)))
        let e = apply(
            apply(
                lambda("x", lambda("y", apply(symbol("y"), symbol("x")))),
                list(&[symbol("+"), integer(2), integer(2)]),
            ),
            lambda("x", list(&[symbol("+"), symbol("x"), integer(1)])),
        );

        let mut env = default_env();

        // => ((\ x (\ y (y x))) 4 (\ x (+ x 1)))
        let e = step_by(e, 4, &mut env);
        let expected = apply(
            apply(
                lambda("x", lambda("y", apply(symbol("y"), symbol("x")))),
                integer(4),
            ),
            lambda("x", list(&[symbol("+"), symbol("x"), integer(1)])),
        );
        assert_eq!(expected, e);

        // => ((\ y (y 4)) (\ x (+ x 1)))
        let e = step_by(e, 1, &mut env);
        let expected = apply(
            lambda("#1", apply(symbol("#1"), integer(4))),
            lambda("x", list(&[symbol("+"), symbol("x"), integer(1)])),
        );
        assert_eq!(expected, e);

        // => (\ x (+ x 1)) 4
        let e = step_by(e, 1, &mut env);
        let expected = apply(
            lambda("x", list(&[symbol("+"), symbol("x"), integer(1)])),
            integer(4),
        );
        assert_eq!(expected, e);

        // => (+ 4 1)
        let e = step_by(e, 1, &mut env);
        let expected = list(&[symbol("+"), integer(4), integer(1)]);
        assert_eq!(expected, e);

        // => 5
        let e = step_by(e, 4, &mut env);
        assert_eq!(e, Exp::Integer(5));
    }
}
