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
    let (name, defines_, macros_) = parser.parse_module().map_err(LoadError::ParseError)?;

    let (mut defines, mut macros) = {
        let module = default_module();
        (module.defines, module.macros)
    };

    for define in defines_.into_iter() {
        if defines.contains_key(&define.name) {
            return Err(LoadError::DuplicateDefinition(define.name));
        }
        defines.insert(define.name.clone(), define);
    }

    for macro_ in macros_.into_iter() {
        if macros.contains_key(&macro_.name) {
            return Err(LoadError::DuplicateDefinition(macro_.name));
        }
        macros.insert(macro_.name.clone(), macro_);
    }

    Ok(Module {
        name,
        defines,
        macros,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Define, Exp};

    #[test]
    fn test_load_module() {
        let source = r#"
        (module test
            (define x () 1)
            (define y () 2))
        "#;
        let module = load_module(source).unwrap();
        assert_eq!(module.defines.len(), default_module().defines.len() + 2);
        assert_eq!(
            module.defines.get("x"),
            Some(&Define::new("x", Exp::Integer(1)))
        );
        assert_eq!(
            module.defines.get("y"),
            Some(&Define::new("y", Exp::Integer(2)))
        );
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
