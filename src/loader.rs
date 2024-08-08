use crate::{
    ast::Module,
    buildin::default_module,
    parser::{ParseError, Parser},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadError {
    DuplicateDefinition(String),
    ParseError(ParseError),
}

pub type Result<T> = std::result::Result<T, LoadError>;

pub fn load_module(source: &str) -> Result<Module> {
    let mut parser = Parser::new(source);
    let module = parser.parse_module().map_err(LoadError::ParseError)?;

    let (mut defines, mut macros) = {
        let module = default_module();
        (module.defines, module.macros)
    };

    for (name, exp) in module.1.into_iter() {
        if defines.contains_key(&name) {
            return Err(LoadError::DuplicateDefinition(name));
        }
        defines.insert(name, exp);
    }

    for (name, exp, args_count) in module.2.into_iter() {
        if macros.contains_key(&name) {
            return Err(LoadError::DuplicateDefinition(name));
        }
        macros.insert(name, (exp, args_count));
    }

    Ok(Module {
        name: module.0,
        defines,
        macros,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Exp;

    #[test]
    fn test_load_module() {
        let source = r#"
        (module test
            (define x () 1)
            (define y () 2))
        "#;
        let module = load_module(source).unwrap();
        assert_eq!(module.defines.len(), default_module().defines.len() + 2);
        assert_eq!(module.defines.get("x"), Some(&Exp::Integer(1)));
        assert_eq!(module.defines.get("y"), Some(&Exp::Integer(2)));
    }

    #[test]
    fn test_load_error_duplicate_definition() {
        let source = r#"
        (module test
            (define x () 1)
            (define x () 2))
        "#;
        let err = load_module(source).unwrap_err();
        assert_eq!(err, LoadError::DuplicateDefinition("x".to_string()));
    }
}
