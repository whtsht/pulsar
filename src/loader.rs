use crate::{
    ast::{Define, Macro, Module, Symbol, UnresolvedModule},
    buildin::default_module,
    parser::{ParseError, Parser},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadError {
    DuplicateDefinition(String),
    ParseError(ParseError),
}

pub type Result<T> = std::result::Result<T, LoadError>;

pub fn load_module(source: &str, module_name: &str) -> Result<Module> {
    let mut parser = Parser::new(source);
    let unresolved_module = parser
        .parse_module(module_name)
        .map_err(LoadError::ParseError)?;
    resolve_module(unresolved_module)
}

pub fn resolve_module(module: UnresolvedModule) -> Result<Module> {
    let (mut defines, mut macros) = {
        let module = default_module();
        (module.defines, module.macros)
    };

    for define in module.defines.into_iter() {
        if defines.contains_key(&define.name) {
            return Err(LoadError::DuplicateDefinition(define.name));
        }
        defines.insert(define.name.clone(), define);
    }

    for macro_ in module.macros.into_iter() {
        if macros.contains_key(&macro_.name) {
            return Err(LoadError::DuplicateDefinition(macro_.name));
        }
        macros.insert(macro_.name.clone(), macro_);
    }

    let inner_modules = module
        .inner_modules
        .into_iter()
        .map(resolve_module)
        .collect::<Result<Vec<_>>>()?;

    Ok(Module {
        name: module.name,
        defines,
        macros,
        inner_modules,
    })
}

impl Module {
    pub fn set_define(&mut self, define: Define) {
        self.defines.insert(define.name.clone(), define);
    }

    pub fn get_define(&self, sym: &Symbol) -> Option<&Define> {
        let mut module = self;
        for name in sym.namespace.iter() {
            module = module.inner_modules.iter().find(|m| &m.name == name)?;
        }
        module.defines.get(&sym.name)
    }

    pub fn set_macro(&mut self, macro_: Macro) {
        self.macros.insert(macro_.name.clone(), macro_);
    }

    pub fn get_macro(&self, sym: &Symbol) -> Option<&Macro> {
        let mut module = self;
        for name in sym.namespace.iter() {
            module = module.inner_modules.iter().find(|m| &m.name == name)?;
        }
        module.macros.get(&sym.name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{list, symbol, Define, Exp};

    #[test]
    fn test_load_module() {
        let func = |exp: Exp| list(&vec![symbol("block", vec![]), exp]);
        let source = r#"
            (define x () 1)
            (define y () 2)
        "#;
        let module = load_module(source, "test").unwrap();
        assert_eq!(module.defines.len(), default_module().defines.len() + 2);
        assert_eq!(
            module.defines.get("x"),
            Some(&Define::new("x", func(Exp::Integer(1))))
        );
        assert_eq!(
            module.defines.get("y"),
            Some(&Define::new("y", func(Exp::Integer(2))))
        );
    }

    #[test]
    fn test_load_error_duplicate_definition() {
        let source = r#"
            (define x () 1)
            (define x () 2)
        "#;
        let err = load_module(source, "test").unwrap_err();
        assert_eq!(err, LoadError::DuplicateDefinition("x".to_string()));
    }

    #[test]
    fn test_inner_module() {
        let source = r#"
            (module test1
                (define x () 1)
            )
            (module test2
                (define y () 2)
            )
        "#;
        let module = load_module(source, "root").unwrap();
        assert_eq!(module.inner_modules.len(), 2);
        assert_eq!(module.inner_modules[0].name, "test1");
        assert_eq!(module.inner_modules[1].name, "test2");
    }
}
