use crate::{
    ast::{self, Exp},
    eval::{eval_with_env, Env, EvalError},
};

fn parse_unary(args: &[Exp], env: &mut Env) -> Result<Exp, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::InvalidArgs);
    }

    let exp = eval_with_env(args[0].clone(), env)?;
    Ok(exp)
}

fn parse_binary(args: &[Exp], env: &mut Env) -> Result<(Exp, Exp), EvalError> {
    if args.len() != 2 {
        return Err(EvalError::InvalidArgs);
    }
    let lhs = eval_with_env(args[0].clone(), env)?;
    let rhs = eval_with_env(args[1].clone(), env)?;
    Ok((lhs, rhs))
}

fn parse_binary_integer(args: &[Exp], env: &mut Env) -> Result<(i64, i64), EvalError> {
    if args.len() != 2 {
        return Err(EvalError::InvalidArgs);
    }
    let lhs = eval_with_env(args[0].clone(), env)?;
    let rhs = eval_with_env(args[1].clone(), env)?;
    Ok((
        lhs.as_integer().ok_or(EvalError::InvalidArgs)?,
        rhs.as_integer().ok_or(EvalError::InvalidArgs)?,
    ))
}

fn add(args: &[Exp], env: &mut Env) -> Result<Exp, EvalError> {
    let (lhs, rhs) = parse_binary_integer(args, env)?;
    Ok(Exp::Integer(lhs + rhs))
}

fn sub(args: &[Exp], env: &mut Env) -> Result<Exp, EvalError> {
    let (lhs, rhs) = parse_binary_integer(args, env)?;
    Ok(Exp::Integer(lhs - rhs))
}

fn mul(args: &[Exp], env: &mut Env) -> Result<Exp, EvalError> {
    let (lhs, rhs) = parse_binary_integer(args, env)?;
    Ok(Exp::Integer(lhs * rhs))
}

fn div(args: &[Exp], env: &mut Env) -> Result<Exp, EvalError> {
    let (lhs, rhs) = parse_binary_integer(args, env)?;
    if rhs == 0 {
        return Err(EvalError::DivideByZero);
    }
    Ok(Exp::Integer(lhs / rhs))
}

fn eq(args: &[Exp], env: &mut Env) -> Result<Exp, EvalError> {
    let (lhs, rhs) = parse_binary(args, env)?;
    Ok(Exp::Bool(lhs == rhs))
}

fn ne(args: &[Exp], env: &mut Env) -> Result<Exp, EvalError> {
    let (lhs, rhs) = parse_binary(args, env)?;
    Ok(Exp::Bool(lhs != rhs))
}

fn cons(args: &[Exp], env: &mut Env) -> Result<Exp, EvalError> {
    let (lhs, rhs) = parse_binary(args, env)?;
    let mut list = vec![lhs];
    list.extend(rhs.clone().as_list().unwrap_or(&[rhs]).iter().cloned());
    Ok(Exp::List(list))
}

fn list(args: &[Exp], _: &mut Env) -> Result<Exp, EvalError> {
    Ok(Exp::List(args.to_vec()))
}

fn first(args: &[Exp], env: &mut Env) -> Result<Exp, EvalError> {
    let exp = parse_unary(args, env)?;
    exp.as_list()
        .and_then(|list| list.get(0).cloned())
        .ok_or(EvalError::InvalidArgs)
}

fn second(args: &[Exp], env: &mut Env) -> Result<Exp, EvalError> {
    let exp = parse_unary(args, env)?;
    exp.as_list()
        .and_then(|list| list.get(1).cloned())
        .ok_or(EvalError::InvalidArgs)
}

fn third(args: &[Exp], env: &mut Env) -> Result<Exp, EvalError> {
    let exp = parse_unary(args, env)?;
    exp.as_list()
        .and_then(|list| list.get(2).cloned())
        .ok_or(EvalError::InvalidArgs)
}

fn nth(args: &[Exp], env: &mut Env) -> Result<Exp, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::InvalidArgs);
    }
    let n = eval_with_env(args[0].clone(), env)?
        .as_integer()
        .ok_or(EvalError::InvalidArgs)?;
    let list = eval_with_env(args[1].clone(), env)?
        .as_list()
        .map(|l| l.to_vec())
        .ok_or(EvalError::InvalidArgs)?;
    list.get(n as usize).cloned().ok_or(EvalError::InvalidArgs)
}

fn is_atom(args: &[Exp], env: &mut Env) -> Result<Exp, EvalError> {
    let exp = parse_unary(args, env)?;
    Ok(ast::bool(match exp {
        Exp::List(_) => false,
        _ => true,
    }))
}

fn print(args: &[Exp], env: &mut Env) -> Result<Exp, EvalError> {
    let exp = parse_unary(args, env)?;
    print!("{} ", exp);
    Ok(Exp::Nil)
}

fn println(args: &[Exp], env: &mut Env) -> Result<Exp, EvalError> {
    let exp = parse_unary(args, env)?;
    println!("{}", exp);
    Ok(Exp::Nil)
}

fn insert_binary_curry_op(
    func: fn(&[Exp], &mut Env) -> Result<Exp, EvalError>,
    func_name: &str,
    env: &mut Env,
) {
    env.insert(
        func_name.to_string(),
        ast::lambda(
            "x",
            ast::lambda(
                "y",
                ast::list(&[Exp::BuildIn(func), ast::symbol("x"), ast::symbol("y")]),
            ),
        ),
    );
}

fn insert_buildin(
    func: fn(&[Exp], &mut Env) -> Result<Exp, EvalError>,
    func_name: &str,
    env: &mut Env,
) {
    env.insert(func_name.to_string(), ast::buildin(func));
}

pub fn default_env() -> Env {
    let mut env = Env::new();

    insert_binary_curry_op(add, "+", &mut env);
    insert_binary_curry_op(sub, "-", &mut env);
    insert_binary_curry_op(mul, "*", &mut env);
    insert_binary_curry_op(div, "/", &mut env);

    insert_binary_curry_op(eq, "==", &mut env);
    insert_binary_curry_op(ne, "/=", &mut env);

    insert_binary_curry_op(cons, "cons", &mut env);
    insert_buildin(list, "list", &mut env);
    insert_buildin(is_atom, "atom?", &mut env);

    insert_buildin(first, "first", &mut env);
    insert_buildin(second, "second", &mut env);
    insert_buildin(third, "third", &mut env);
    insert_binary_curry_op(nth, "nth", &mut env);

    insert_buildin(print, "print", &mut env);
    insert_buildin(println, "println", &mut env);

    env
}

#[cfg(test)]
mod tests {
    use crate::{ast::*, eval::eval};

    #[test]
    fn test_integer_binary_op() {
        // (+ 1 2)
        let e = list(&[symbol("+"), integer(1), integer(2)]);
        assert_eq!(eval(e), Ok(Exp::Integer(3)));

        // (- 1 2)
        let e = list(&[symbol("-"), integer(1), integer(2)]);
        assert_eq!(eval(e), Ok(Exp::Integer(-1)));
    }

    #[test]
    fn test_compare_op() {
        // (== 1 1) => true
        let e = list(&[symbol("=="), integer(1), integer(1)]);
        assert_eq!(eval(e), Ok(bool(true)));

        // (/= 1 1) => false
        let e = list(&[symbol("/="), integer(1), integer(1)]);
        assert_eq!(eval(e), Ok(bool(false)));

        // (/= '(1 2) 2) => true
        let e = list(&[
            symbol("/="),
            quote(list(&[integer(1), integer(2)])),
            integer(2),
        ]);
        assert_eq!(eval(e), Ok(bool(true)));

        // (== '(1 2) 2) => false
        let e = list(&[
            symbol("=="),
            quote(list(&[integer(1), integer(2)])),
            integer(2),
        ]);
        assert_eq!(eval(e), Ok(bool(false)));
    }

    #[test]
    fn test_cons() {
        // (cons 1 '(2 3)) => (1 2 3)
        let e = list(&[
            symbol("cons"),
            integer(1),
            quote(list(&[integer(2), integer(3)])),
        ]);
        assert_eq!(eval(e), Ok(list(&[integer(1), integer(2), integer(3)])));

        // (cons 1 2) => (1 2)
        let e = list(&[symbol("cons"), integer(1), integer(2)]);
        assert_eq!(eval(e), Ok(list(&[integer(1), integer(2)])));

        // (cons '(1 2) 3) => ((1 2) 3)
        let e = list(&[
            symbol("cons"),
            quote(list(&[integer(1), integer(2)])),
            integer(3),
        ]);
        assert_eq!(
            eval(e),
            Ok(list(&[list(&[integer(1), integer(2)]), integer(3)]))
        );
    }

    #[test]
    fn test_list() {
        // (list 1 2 3) => (1 2 3)
        let e = list(&[symbol("list"), integer(1), integer(2), integer(3)]);
        assert_eq!(eval(e), Ok(list(&[integer(1), integer(2), integer(3)])));
    }

    #[test]
    fn test_is_atom() {
        // (atom? 1) => true
        let e = list(&[symbol("atom?"), integer(1)]);
        assert_eq!(eval(e), Ok(bool(true)));

        // (atom? '(1 2)) => false
        let e = list(&[symbol("atom?"), quote(list(&[integer(1), integer(2)]))]);
        assert_eq!(eval(e), Ok(bool(false)));
    }

    #[test]
    fn test_nth() {
        // (first '(1 2)) => 1
        let e = list(&[symbol("first"), quote(list(&[integer(1), integer(2)]))]);
        assert_eq!(eval(e), Ok(integer(1)));

        // (second '(1 2)) => 2
        let e = list(&[symbol("second"), quote(list(&[integer(1), integer(2)]))]);
        assert_eq!(eval(e), Ok(integer(2)));

        // (third '(1 2 3)) => 3
        let e = list(&[
            symbol("third"),
            quote(list(&[integer(1), integer(2), integer(3)])),
        ]);
        assert_eq!(eval(e), Ok(integer(3)));

        // (nth 5 '(1 2 3 4 5 6 7)) => 6
        let e = list(&[
            symbol("nth"),
            integer(5),
            quote(list(&[
                integer(1),
                integer(2),
                integer(3),
                integer(4),
                integer(5),
                integer(6),
                integer(7),
            ])),
        ]);
        assert_eq!(eval(e), Ok(integer(6)));
    }
}
