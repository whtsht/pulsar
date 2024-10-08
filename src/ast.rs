use std::{collections::HashMap, fmt::Display};

use crate::eval::{EvalError, VariableGenerator};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Module {
    pub name: String,
    pub defines: HashMap<String, Define>,
    pub macros: HashMap<String, Macro>,
    pub inner_modules: Vec<Module>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct UnresolvedModule {
    pub name: String,
    pub defines: Vec<Define>,
    pub macros: Vec<Macro>,
    pub inner_modules: Vec<UnresolvedModule>,
    pub imported: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Define {
    pub name: String,
    pub exp: Exp,
}

impl Define {
    pub fn new(name: &str, exp: Exp) -> Self {
        Define {
            name: name.to_string(),
            exp,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Macro {
    pub name: String,
    pub exp: Exp,
    pub args: Vec<String>,
    pub var_arg: Option<String>,
}

impl Macro {
    pub fn new(name: &str, exp: Exp, args: Vec<String>, var_arg: Option<String>) -> Self {
        Macro {
            name: name.to_string(),
            exp,
            args,
            var_arg,
        }
    }
}

impl Module {
    pub fn new(name: &str) -> Self {
        Module {
            name: name.to_string(),
            defines: HashMap::new(),
            macros: HashMap::new(),
            inner_modules: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Symbol {
    pub name: String,
    pub namespace: Vec<String>,
}

impl Symbol {
    pub fn new(name: String, namespace: Vec<String>) -> Self {
        Symbol {
            name: name.to_string(),
            namespace,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Exp {
    Nil,
    Bool(bool),
    Integer(i64),
    String(String),
    Symbol(Symbol),
    Lambda(String, Box<Exp>),
    Apply(Box<Exp>, Box<Exp>),
    List(Vec<Exp>),
    If(Box<Exp>, Box<Exp>, Box<Exp>),
    Quote(Box<Exp>),
    BackQuote(Box<Exp>),
    UnQuote(Box<Exp>),
    Extend(Box<Exp>),
    Let((String, Box<Exp>), Box<Exp>),
    BuildIn(fn(&[Exp], &Module, &mut VariableGenerator) -> Result<Exp, EvalError>),
}

impl Exp {
    pub fn as_nil(&self) -> Option<()> {
        match self {
            Exp::Nil => Some(()),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Exp::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Exp::Integer(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&str> {
        match self {
            Exp::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_symbol(&self) -> Option<&Symbol> {
        match self {
            Exp::Symbol(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_list(&self) -> Option<&[Exp]> {
        match self {
            Exp::List(l) => Some(l),
            _ => None,
        }
    }
}

impl Display for Exp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Exp::Nil => write!(f, "nil"),
            Exp::Bool(bool) => write!(f, "{}", bool),
            Exp::Integer(integer) => write!(f, "{}", integer),
            Exp::String(str) => write!(f, "{}", str),
            Exp::Symbol(sym) => write!(f, "{}", sym.name),
            Exp::Lambda(arg, exp) => write!(f, "(\\ ({}) {})", arg, exp),
            Exp::Apply(exp1, exp2) => write!(f, "({} {})", exp1, exp2),
            Exp::List(exps) => write!(
                f,
                "({})",
                exps.iter()
                    .map(|exp| exp.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
            Exp::If(cond, then, else_) => write!(f, "(if {} {} {})", cond, then, else_),
            Exp::Quote(exp) => write!(f, "'{}", exp),
            Exp::BackQuote(exp) => write!(f, "`{}", exp),
            Exp::UnQuote(exp) => write!(f, ".{}", exp),
            Exp::Extend(exp) => write!(f, "@{}", exp),
            Exp::Let((bind, exp1), exp2) => write!(f, "(let ({} {}) {})", bind, exp1, exp2),
            Exp::BuildIn(_) => write!(f, "#buildin",),
        }
    }
}

pub fn nil() -> Exp {
    Exp::Nil
}

pub fn bool(b: bool) -> Exp {
    Exp::Bool(b)
}

pub fn integer(i: i64) -> Exp {
    Exp::Integer(i)
}

pub fn string(s: &str) -> Exp {
    Exp::String(s.to_string())
}

pub fn symbol(name: &str, namespace: Vec<String>) -> Exp {
    Exp::Symbol(Symbol {
        name: name.to_string(),
        namespace,
    })
}

pub fn lambda(param: &str, body: Exp) -> Exp {
    Exp::Lambda(param.to_string(), Box::new(body))
}

pub fn apply(e1: Exp, e2: Exp) -> Exp {
    Exp::Apply(Box::new(e1), Box::new(e2))
}

pub fn list(list: &[Exp]) -> Exp {
    Exp::List(list.to_vec())
}

pub fn if_(cond: Exp, then: Exp, else_: Exp) -> Exp {
    Exp::If(Box::new(cond), Box::new(then), Box::new(else_))
}

pub fn let_(bind: (&str, Exp), exp: Exp) -> Exp {
    Exp::Let((bind.0.to_string(), Box::new(bind.1)), Box::new(exp))
}

pub fn quote(e: Exp) -> Exp {
    Exp::Quote(Box::new(e))
}

pub fn backquote(e: Exp) -> Exp {
    Exp::BackQuote(Box::new(e))
}

pub fn unquote(e: Exp) -> Exp {
    Exp::UnQuote(Box::new(e))
}

pub fn extend(e: Exp) -> Exp {
    Exp::Extend(Box::new(e))
}

pub fn buildin(f: fn(&[Exp], &Module, &mut VariableGenerator) -> Result<Exp, EvalError>) -> Exp {
    Exp::BuildIn(f)
}
