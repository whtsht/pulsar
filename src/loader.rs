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

pub fn load_module(name: &str, source: &str) -> Result<Module> {
    let mut parser = Parser::new(source);
    let defs = parser
        .parse_defines()
        .map_err(|err| LoadError::ParseError(err))?;

    let mut defines = default_module().defines;
    for (name, exp) in defs.into_iter() {
        if defines.contains_key(&name) {
            return Err(LoadError::DuplicateDefinition(name));
        }
        defines.insert(name, exp);
    }

    Ok(Module {
        name: name.to_string(),
        defines,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Exp;

    #[test]
    fn test_load_module() {
        let source = r#"
        (define x () 1)
        (define y () 2)
        "#;
        let module = load_module("test", source).unwrap();
        assert_eq!(module.defines.len(), default_module().defines.len() + 2);
        assert_eq!(module.defines.get("x"), Some(&Exp::Integer(1)));
        assert_eq!(module.defines.get("y"), Some(&Exp::Integer(2)));
    }

    #[test]
    fn test_load_error_duplicate_definition() {
        let source = r#"
        (define x () 1)
        (define x () 2)
        "#;
        let err = load_module("test", source).unwrap_err();
        assert_eq!(err, LoadError::DuplicateDefinition("x".to_string()));
    }
}
