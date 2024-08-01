use std::io::{self, Write};
use topogi_lang::{
    eval,
    parser::{parse_error_message, Parser},
};

fn main() {
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

        let ast = match Parser::new(input).parse_exp() {
            Ok(ast) => ast,
            Err(err) => {
                println!("{}", parse_error_message(err, input));
                continue;
            }
        };

        match eval::eval_default_module(ast) {
            Ok(result) => println!("\n==> {}", result),
            Err(e) => {
                println!("{:?}", e);
                continue;
            }
        }
    }
}
