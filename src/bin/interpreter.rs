use pulsar::{
    buildin::default_module,
    parser::{parse_error_message, Parser},
};
use std::io::{self, Write};

fn main() {
    let mut module = default_module();

    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line");

        let input = input.trim();

        if input == "exit" {
            break;
        }

        match Parser::new(input).parse_defines_or_macros() {
            Ok((defines, macros)) => {
                for (name, exp) in defines {
                    module.defines.insert(name, exp);
                }
                for (name, exp, args) in macros {
                    module.macros.insert(name, (exp, args));
                }
                continue;
            }
            Err(_) => {}
        }

        let exp = match Parser::new(input).parse_exp() {
            Ok(ast) => ast,
            Err(err) => {
                println!("{}", parse_error_message(err, input));
                continue;
            }
        };

        match module.eval(exp) {
            Ok(result) => println!("=> {}\n", result),
            Err(e) => {
                println!("{:?}", e);
                continue;
            }
        }
    }
}
