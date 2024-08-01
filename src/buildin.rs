use crate::{
    ast::{self, Exp, Module},
    eval::{eval, EvalError, VariableGenerator},
};

fn parse_unary(
    args: &[Exp],
    module: &Module,
    gen: &mut VariableGenerator,
) -> Result<Exp, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::InvalidArgs);
    }

    let exp = eval(args[0].clone(), module, gen)?;
    Ok(exp)
}

fn parse_binary(
    args: &[Exp],
    module: &Module,
    gen: &mut VariableGenerator,
) -> Result<(Exp, Exp), EvalError> {
    if args.len() != 2 {
        return Err(EvalError::InvalidArgs);
    }
    let lhs = eval(args[0].clone(), module, gen)?;
    let rhs = eval(args[1].clone(), module, gen)?;
    Ok((lhs, rhs))
}

fn parse_binary_integer(
    args: &[Exp],
    module: &Module,
    gen: &mut VariableGenerator,
) -> Result<(i64, i64), EvalError> {
    if args.len() != 2 {
        return Err(EvalError::InvalidArgs);
    }
    let lhs = eval(args[0].clone(), module, gen)?;
    let rhs = eval(args[1].clone(), module, gen)?;
    Ok((
        lhs.as_integer().ok_or(EvalError::InvalidArgs)?,
        rhs.as_integer().ok_or(EvalError::InvalidArgs)?,
    ))
}

fn add(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp, EvalError> {
    let (lhs, rhs) = parse_binary_integer(args, module, gen)?;
    Ok(Exp::Integer(lhs + rhs))
}

fn sub(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp, EvalError> {
    let (lhs, rhs) = parse_binary_integer(args, module, gen)?;
    Ok(Exp::Integer(lhs - rhs))
}

fn mul(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp, EvalError> {
    let (lhs, rhs) = parse_binary_integer(args, module, gen)?;
    Ok(Exp::Integer(lhs * rhs))
}

fn div(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp, EvalError> {
    let (lhs, rhs) = parse_binary_integer(args, module, gen)?;
    if rhs == 0 {
        return Err(EvalError::DivideByZero);
    }
    Ok(Exp::Integer(lhs / rhs))
}

fn eq(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp, EvalError> {
    let (lhs, rhs) = parse_binary(args, module, gen)?;
    Ok(Exp::Bool(lhs == rhs))
}

fn ne(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp, EvalError> {
    let (lhs, rhs) = parse_binary(args, module, gen)?;
    Ok(Exp::Bool(lhs != rhs))
}

fn cons(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp, EvalError> {
    let (lhs, rhs) = parse_binary(args, module, gen)?;
    let mut list = vec![lhs];
    list.extend(rhs.clone().as_list().unwrap_or(&[rhs]).iter().cloned());
    Ok(Exp::List(list))
}

fn list(args: &[Exp], _: &Module, _: &mut VariableGenerator) -> Result<Exp, EvalError> {
    Ok(Exp::List(args.to_vec()))
}

fn first(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp, EvalError> {
    let exp = parse_unary(args, module, gen)?;
    exp.as_list()
        .and_then(|list| list.get(0).cloned())
        .ok_or(EvalError::InvalidArgs)
}

fn second(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp, EvalError> {
    let exp = parse_unary(args, module, gen)?;
    exp.as_list()
        .and_then(|list| list.get(1).cloned())
        .ok_or(EvalError::InvalidArgs)
}

fn third(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp, EvalError> {
    let exp = parse_unary(args, module, gen)?;
    exp.as_list()
        .and_then(|list| list.get(2).cloned())
        .ok_or(EvalError::InvalidArgs)
}

fn nth(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::InvalidArgs);
    }
    let n = eval(args[0].clone(), module, gen)?
        .as_integer()
        .ok_or(EvalError::InvalidArgs)?;
    let list = eval(args[1].clone(), module, gen)?
        .as_list()
        .map(|l| l.to_vec())
        .ok_or(EvalError::InvalidArgs)?;
    list.get(n as usize).cloned().ok_or(EvalError::InvalidArgs)
}

fn is_atom(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp, EvalError> {
    let exp = parse_unary(args, module, gen)?;
    Ok(ast::bool(match exp {
        Exp::List(_) => false,
        _ => true,
    }))
}

fn print(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp, EvalError> {
    let exp = parse_unary(args, module, gen)?;
    print!("{} ", exp);
    Ok(Exp::Nil)
}

fn println(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp, EvalError> {
    let exp = parse_unary(args, module, gen)?;
    println!("{}", exp);
    Ok(Exp::Nil)
}

fn insert_binary_curry_op(
    func: fn(&[Exp], &Module, &mut VariableGenerator) -> Result<Exp, EvalError>,
    func_name: &str,
    module: &mut Module,
) {
    module.defines.insert(
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
    func: fn(&[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp, EvalError>,
    func_name: &str,
    module: &mut Module,
) {
    module
        .defines
        .insert(func_name.to_string(), ast::buildin(func));
}

pub fn default_module() -> Module {
    let mut module = Module::new("##default##");

    insert_binary_curry_op(add, "+", &mut module);
    insert_binary_curry_op(sub, "-", &mut module);
    insert_binary_curry_op(mul, "*", &mut module);
    insert_binary_curry_op(div, "/", &mut module);

    insert_binary_curry_op(eq, "==", &mut module);
    insert_binary_curry_op(ne, "/=", &mut module);

    insert_binary_curry_op(cons, "cons", &mut module);
    insert_buildin(list, "list", &mut module);
    insert_buildin(is_atom, "atom?", &mut module);

    insert_buildin(first, "first", &mut module);
    insert_buildin(second, "second", &mut module);
    insert_buildin(third, "third", &mut module);
    insert_binary_curry_op(nth, "nth", &mut module);

    insert_buildin(print, "print", &mut module);
    insert_buildin(println, "println", &mut module);

    module
}

#[cfg(test)]
mod tests {
    use crate::{ast::*, eval::eval_default_module};

    #[test]
    fn test_integer_binary_op() {
        // (+ 1 2)
        let e = list(&[symbol("+"), integer(1), integer(2)]);
        assert_eq!(eval_default_module(e), Ok(Exp::Integer(3)));

        // (- 1 2)
        let e = list(&[symbol("-"), integer(1), integer(2)]);
        assert_eq!(eval_default_module(e), Ok(Exp::Integer(-1)));
    }

    #[test]
    fn test_compare_op() {
        // (== 1 1) => true
        let e = list(&[symbol("=="), integer(1), integer(1)]);
        assert_eq!(eval_default_module(e), Ok(bool(true)));

        // (/= 1 1) => false
        let e = list(&[symbol("/="), integer(1), integer(1)]);
        assert_eq!(eval_default_module(e), Ok(bool(false)));

        // (/= '(1 2) 2) => true
        let e = list(&[
            symbol("/="),
            quote(list(&[integer(1), integer(2)])),
            integer(2),
        ]);
        assert_eq!(eval_default_module(e), Ok(bool(true)));

        // (== '(1 2) 2) => false
        let e = list(&[
            symbol("=="),
            quote(list(&[integer(1), integer(2)])),
            integer(2),
        ]);
        assert_eq!(eval_default_module(e), Ok(bool(false)));
    }

    #[test]
    fn test_cons() {
        // (cons 1 '(2 3)) => (1 2 3)
        let e = list(&[
            symbol("cons"),
            integer(1),
            quote(list(&[integer(2), integer(3)])),
        ]);
        assert_eq!(
            eval_default_module(e),
            Ok(list(&[integer(1), integer(2), integer(3)]))
        );

        // (cons 1 2) => (1 2)
        let e = list(&[symbol("cons"), integer(1), integer(2)]);
        assert_eq!(eval_default_module(e), Ok(list(&[integer(1), integer(2)])));

        // (cons '(1 2) 3) => ((1 2) 3)
        let e = list(&[
            symbol("cons"),
            quote(list(&[integer(1), integer(2)])),
            integer(3),
        ]);
        assert_eq!(
            eval_default_module(e),
            Ok(list(&[list(&[integer(1), integer(2)]), integer(3)]))
        );
    }

    #[test]
    fn test_list() {
        // (list 1 2 3) => (1 2 3)
        let e = list(&[symbol("list"), integer(1), integer(2), integer(3)]);
        assert_eq!(
            eval_default_module(e),
            Ok(list(&[integer(1), integer(2), integer(3)]))
        );
    }

    #[test]
    fn test_is_atom() {
        // (atom? 1) => true
        let e = list(&[symbol("atom?"), integer(1)]);
        assert_eq!(eval_default_module(e), Ok(bool(true)));

        // (atom? '(1 2)) => false
        let e = list(&[symbol("atom?"), quote(list(&[integer(1), integer(2)]))]);
        assert_eq!(eval_default_module(e), Ok(bool(false)));
    }

    #[test]
    fn test_nth() {
        // (first '(1 2)) => 1
        let e = list(&[symbol("first"), quote(list(&[integer(1), integer(2)]))]);
        assert_eq!(eval_default_module(e), Ok(integer(1)));

        // (second '(1 2)) => 2
        let e = list(&[symbol("second"), quote(list(&[integer(1), integer(2)]))]);
        assert_eq!(eval_default_module(e), Ok(integer(2)));

        // (third '(1 2 3)) => 3
        let e = list(&[
            symbol("third"),
            quote(list(&[integer(1), integer(2), integer(3)])),
        ]);
        assert_eq!(eval_default_module(e), Ok(integer(3)));

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
        assert_eq!(eval_default_module(e), Ok(integer(6)));
    }
}
