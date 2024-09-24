use std::{
    env,
    fs::File,
    io::{BufWriter, Write as _},
    path::Path,
};

const HEADERS: &str = include_str!("data/headers.txt");

fn main() {
    let path = Path::new(&env::var("OUT_DIR").unwrap()).join("codegen.rs");
    let mut file = BufWriter::new(File::create(path).unwrap());
    let mut headers_set = phf_codegen::Set::<&str>::new();
    for h in HEADERS.lines() {
        headers_set.entry(h);
    }
    write!(
        &mut file,
        "static HEADERS_PHF: phf::Set<&str> = {};",
        headers_set.build()
    )
    .unwrap();
}
