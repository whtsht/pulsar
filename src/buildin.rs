use std::ops::Not;

use crate::{
    ast::{self, apply, Exp, Module},
    eval::{eval, EvalError, Result, VariableGenerator},
};

fn parse_unary(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    if args.len() != 1 {
        return Err(EvalError::InvalidArgs(args.to_vec()));
    }

    let exp = eval(args[0].clone(), module, gen)?;
    Ok(exp)
}

fn parse_binary(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<(Exp, Exp)> {
    if args.len() != 2 {
        return Err(EvalError::InvalidArgs(args.to_vec()));
    }
    let lhs = eval(args[0].clone(), module, gen)?;
    let rhs = eval(args[1].clone(), module, gen)?;
    Ok((lhs, rhs))
}

fn parse_ternary(
    args: &[Exp],
    module: &Module,
    gen: &mut VariableGenerator,
) -> Result<(Exp, Exp, Exp)> {
    if args.len() != 3 {
        return Err(EvalError::InvalidArgs(args.to_vec()));
    }
    let lhs = eval(args[0].clone(), module, gen)?;
    let mhs = eval(args[1].clone(), module, gen)?;
    let rhs = eval(args[2].clone(), module, gen)?;
    Ok((lhs, mhs, rhs))
}

fn parse_binary_integer(
    args: &[Exp],
    module: &Module,
    gen: &mut VariableGenerator,
) -> Result<(i64, i64)> {
    if args.len() != 2 {
        return Err(EvalError::InvalidArgs(args.to_vec()));
    }
    let lhs = eval(args[0].clone(), module, gen)?;
    let rhs = eval(args[1].clone(), module, gen)?;
    Ok((
        lhs.as_integer()
            .ok_or(EvalError::InvalidArgs(args.to_vec()))?,
        rhs.as_integer()
            .ok_or(EvalError::InvalidArgs(args.to_vec()))?,
    ))
}

fn add(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    let (lhs, rhs) = parse_binary_integer(args, module, gen)?;
    Ok(Exp::Integer(lhs + rhs))
}

fn sub(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    let (lhs, rhs) = parse_binary_integer(args, module, gen)?;
    Ok(Exp::Integer(lhs - rhs))
}

fn mul(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    let (lhs, rhs) = parse_binary_integer(args, module, gen)?;
    Ok(Exp::Integer(lhs * rhs))
}

fn div(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    let (lhs, rhs) = parse_binary_integer(args, module, gen)?;
    if rhs == 0 {
        return Err(EvalError::DivideByZero(apply(
            args[0].clone(),
            args[1].clone(),
        )));
    }
    Ok(Exp::Integer(lhs / rhs))
}

fn odd(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    let int = parse_unary(args, module, gen)?
        .as_integer()
        .ok_or(EvalError::InvalidArgs(args.to_vec()))?;

    Ok(Exp::Bool(int % 2 != 0))
}

fn even(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    let int = parse_unary(args, module, gen)?
        .as_integer()
        .ok_or(EvalError::InvalidArgs(args.to_vec()))?;

    Ok(Exp::Bool(int % 2 == 0))
}

fn eq(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    let (lhs, rhs) = parse_binary(args, module, gen)?;
    Ok(Exp::Bool(lhs == rhs))
}

fn ne(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    let (lhs, rhs) = parse_binary(args, module, gen)?;
    Ok(Exp::Bool(lhs != rhs))
}

fn cons(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    let (lhs, rhs) = parse_binary(args, module, gen)?;
    let mut list = vec![lhs];
    list.extend(rhs.clone().as_list().unwrap_or(&[rhs]).iter().cloned());
    Ok(Exp::List(list))
}

fn list(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    let args = args
        .iter()
        .cloned()
        .map(|exp| eval(exp, module, gen))
        .collect::<Result<_>>()?;
    Ok(Exp::List(args))
}

fn first(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    let exp = parse_unary(args, module, gen)?;
    exp.as_list()
        .and_then(|list| list.get(0).cloned())
        .ok_or(EvalError::InvalidArgs(args.to_vec()))
}

fn second(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    let exp = parse_unary(args, module, gen)?;
    exp.as_list()
        .and_then(|list| list.get(1).cloned())
        .ok_or(EvalError::InvalidArgs(args.to_vec()))
}

fn third(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    let exp = parse_unary(args, module, gen)?;
    exp.as_list()
        .and_then(|list| list.get(2).cloned())
        .ok_or(EvalError::InvalidArgs(args.to_vec()))
}

fn nth(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    if args.len() != 2 {
        return Err(EvalError::InvalidArgs(args.to_vec()));
    }
    let n = eval(args[0].clone(), module, gen)?
        .as_integer()
        .ok_or(EvalError::InvalidArgs(args.to_vec()))?;
    let list = eval(args[1].clone(), module, gen)?
        .as_list()
        .map(|l| l.to_vec())
        .ok_or(EvalError::InvalidArgs(args.to_vec()))?;
    list.get(n as usize)
        .cloned()
        .ok_or(EvalError::InvalidArgs(args.to_vec()))
}

fn is_atom(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    let exp = parse_unary(args, module, gen)?;
    Ok(ast::bool(matches!(exp, Exp::List(_)).not()))
}

fn print(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    let exp = parse_unary(args, module, gen)?;
    print!("{} ", exp);
    Ok(Exp::Nil)
}

fn println(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    let exp = parse_unary(args, module, gen)?;
    println!("{}", exp);
    Ok(Exp::Nil)
}

fn string_append(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    let (lhs, rhs) = parse_binary(args, module, gen)?;
    let lhs = lhs
        .as_string()
        .ok_or(EvalError::InvalidArgs(args.to_vec()))?;
    let rhs = rhs
        .as_string()
        .ok_or(EvalError::InvalidArgs(args.to_vec()))?;
    Ok(Exp::String(format!("{}{}", lhs, rhs)))
}

fn string_head(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    let exp = parse_unary(args, module, gen)?;
    let s = exp
        .as_string()
        .ok_or(EvalError::InvalidArgs(args.to_vec()))?;
    Ok(Exp::String(s.chars().take(1).collect()))
}

fn string_tail(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    let exp = parse_unary(args, module, gen)?;
    let s = exp
        .as_string()
        .ok_or(EvalError::InvalidArgs(args.to_vec()))?;
    Ok(Exp::String(s.chars().skip(1).collect()))
}

fn string_init(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    let exp = parse_unary(args, module, gen)?;
    let s = exp
        .as_string()
        .ok_or(EvalError::InvalidArgs(args.to_vec()))?;
    Ok(Exp::String(s.chars().take(s.len() - 1).collect()))
}

fn string_last(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    let exp = parse_unary(args, module, gen)?;
    let s = exp
        .as_string()
        .ok_or(EvalError::InvalidArgs(args.to_vec()))?;
    Ok(Exp::String(s.chars().rev().take(1).collect()))
}

fn symbol_to_string(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    let exp = parse_unary(args, module, gen)?;
    let s = exp
        .as_symbol()
        .ok_or(EvalError::InvalidArgs(args.to_vec()))?;
    Ok(Exp::String(s.to_string()))
}

fn foldr(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    let (f, mut acc, list) = parse_ternary(args, module, gen)?;
    for elem in list
        .as_list()
        .ok_or(EvalError::InvalidArgs(args.to_vec()))?
    {
        acc = eval(
            apply(apply(f.clone(), elem.clone()), acc.clone()),
            module,
            gen,
        )?;
    }
    Ok(acc)
}

fn foldl(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    let (f, mut acc, list) = parse_ternary(args, module, gen)?;
    for elem in list
        .as_list()
        .ok_or(EvalError::InvalidArgs(args.to_vec()))?
        .iter()
        .rev()
        .cloned()
    {
        acc = eval(apply(apply(f.clone(), acc), elem), module, gen)?;
    }
    Ok(acc)
}

fn map(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    let (f, list) = parse_binary(args, module, gen)?;
    let list = list
        .as_list()
        .ok_or(EvalError::InvalidArgs(args.to_vec()))?;
    let mut result = vec![];
    for elem in list.iter().cloned() {
        result.push(eval(apply(f.clone(), elem), module, gen)?);
    }
    Ok(Exp::List(result))
}

fn filter(args: &[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp> {
    let (f, list) = parse_binary(args, module, gen)?;
    let list = list
        .as_list()
        .ok_or(EvalError::InvalidArgs(args.to_vec()))?;
    let mut result = vec![];
    for elem in list.iter().cloned() {
        if eval(apply(f.clone(), elem.clone()), module, gen)?
            .as_bool()
            .ok_or(EvalError::ExpectedBool(elem.clone()))?
        {
            result.push(elem)
        }
    }
    Ok(Exp::List(result))
}

fn insert_binary_curry_op(
    func: fn(&[Exp], &Module, &mut VariableGenerator) -> Result<Exp>,
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

fn insert_ternary_curry_op(
    func: fn(&[Exp], &Module, &mut VariableGenerator) -> Result<Exp>,
    func_name: &str,
    module: &mut Module,
) {
    module.defines.insert(
        func_name.to_string(),
        ast::lambda(
            "x",
            ast::lambda(
                "y",
                ast::lambda(
                    "z",
                    ast::list(&[
                        Exp::BuildIn(func),
                        ast::symbol("x"),
                        ast::symbol("y"),
                        ast::symbol("z"),
                    ]),
                ),
            ),
        ),
    );
}

fn insert_unary_op(
    func: fn(&[Exp], module: &Module, gen: &mut VariableGenerator) -> Result<Exp>,
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
    insert_unary_op(odd, "odd", &mut module);
    insert_unary_op(even, "even", &mut module);

    insert_binary_curry_op(eq, "=", &mut module);
    insert_binary_curry_op(ne, "/=", &mut module);

    insert_binary_curry_op(cons, "cons", &mut module);
    insert_unary_op(list, "list", &mut module);
    insert_unary_op(is_atom, "atom?", &mut module);

    insert_unary_op(first, "first", &mut module);
    insert_unary_op(second, "second", &mut module);
    insert_unary_op(third, "third", &mut module);
    insert_binary_curry_op(nth, "nth", &mut module);

    insert_unary_op(print, "print", &mut module);
    insert_unary_op(println, "println", &mut module);

    insert_binary_curry_op(string_append, "string-append", &mut module);
    insert_unary_op(string_head, "string-head", &mut module);
    insert_unary_op(string_tail, "string-tail", &mut module);
    insert_unary_op(string_init, "string-init", &mut module);
    insert_unary_op(string_last, "string-last", &mut module);

    insert_unary_op(symbol_to_string, "symbol->string", &mut module);

    insert_ternary_curry_op(foldr, "foldr", &mut module);
    insert_ternary_curry_op(foldl, "foldl", &mut module);
    insert_binary_curry_op(map, "map", &mut module);
    insert_binary_curry_op(filter, "filter", &mut module);

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
        // (= 1 1) => true
        let e = list(&[symbol("="), integer(1), integer(1)]);
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

        // (= '(1 2) 2) => false
        let e = list(&[
            symbol("="),
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
        // (list (+ 1 2)) => (3)
        let e = list(&[symbol("list"), list(&[symbol("+"), integer(1), integer(2)])]);
        assert_eq!(eval_default_module(e), Ok(list(&[integer(3)])));
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

    #[test]
    fn test_string_append() {
        // (string-append "abc" "def") => "abcdef"
        let e = list(&[symbol("string-append"), string("abc"), string("def")]);
        assert_eq!(eval_default_module(e), Ok(string("abcdef")));
    }

    #[test]
    fn test_string_head() {
        // (string-head "abc") => "a"
        let e = list(&[symbol("string-head"), string("abc")]);
        assert_eq!(eval_default_module(e), Ok(string("a")));
    }

    #[test]
    fn test_string_tail() {
        // (string-tail "abc") => "bc"
        let e = list(&[symbol("string-tail"), string("abc")]);
        assert_eq!(eval_default_module(e), Ok(string("bc")));
    }

    #[test]
    fn test_string_init() {
        // (string-init "abc") => "ab"
        let e = list(&[symbol("string-init"), string("abc")]);
        assert_eq!(eval_default_module(e), Ok(string("ab")));
    }

    #[test]
    fn test_string_last() {
        // (string-last "abc") => "c"
        let e = list(&[symbol("string-last"), string("abc")]);
        assert_eq!(eval_default_module(e), Ok(string("c")));
    }

    #[test]
    fn test_symbol_to_string() {
        // (symbol->string 'abc) => "abc"
        let e = list(&[symbol("symbol->string"), quote(symbol("abc"))]);
        assert_eq!(eval_default_module(e), Ok(string("abc")));
    }

    #[test]
    fn test_fold() {
        // (foldr - 0 '(1 2 3 4 5)) => 3
        let e = list(&[
            symbol("foldr"),
            symbol("-"),
            integer(0),
            quote(list(&[
                integer(1),
                integer(2),
                integer(3),
                integer(4),
                integer(5),
            ])),
        ]);
        assert_eq!(eval_default_module(e), Ok(integer(3)));

        // (foldl - 0 '(1 2 3 4 5)) => -15
        let e = list(&[
            symbol("foldl"),
            symbol("-"),
            integer(0),
            quote(list(&[
                integer(1),
                integer(2),
                integer(3),
                integer(4),
                integer(5),
            ])),
        ]);
        assert_eq!(eval_default_module(e), Ok(integer(-15)));
    }

    #[test]
    fn test_filter() {
        // (filter odd '(1 2 3 4 5)) => (1 3 5)
        let e = list(&[
            symbol("filter"),
            symbol("odd"),
            quote(list(&[
                integer(1),
                integer(2),
                integer(3),
                integer(4),
                integer(5),
            ])),
        ]);
        assert_eq!(
            eval_default_module(e),
            Ok(list(&[integer(1), integer(3), integer(5)]))
        );
    }
}
