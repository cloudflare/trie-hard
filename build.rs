use std::{
    collections::HashSet,
    env,
    fs::File,
    io::{BufWriter, Write as _},
    path::Path,
};

const HEADERS: &str = include_str!("data/headers.txt");
const SUN_RISING: &str = include_str!("data/sun-rising.txt");

fn main() {
    let path = Path::new(&env::var("OUT_DIR").unwrap()).join("codegen.rs");
    let mut file = BufWriter::new(File::create(path).unwrap());
    let mut headers_set = phf_codegen::Set::<&str>::new();
    let headers_rev: Vec<_> = HEADERS
        .lines()
        .collect::<HashSet<_>>()
        .into_iter()
        .map(|s| s.chars().rev().collect::<String>())
        .collect();
    for s in &headers_rev {
        headers_set.entry(s);
    }
    write!(
        &mut file,
        "static HEADERS_PHF: phf::Set<&str> = {};",
        headers_set.build()
    )
    .unwrap();

    let mut small_set = phf_codegen::Set::<&str>::new();
    for s in SUN_RISING
        .split(|c: char| c.is_whitespace())
        .collect::<HashSet<_>>()
        .into_iter()
    {
        small_set.entry(s);
    }
    write!(
        &mut file,
        "static SMALL_PHF: phf::Set<&str> = {};",
        small_set.build()
    )
    .unwrap();

    let mut big_set = phf_codegen::Set::<&str>::new();
    const OW_1984: &str = include_str!("data/1984.txt");
    for s in OW_1984
        .split(|c: char| c.is_whitespace())
        .collect::<HashSet<_>>()
        .into_iter()
    {
        big_set.entry(s);
    }
    write!(
        &mut file,
        "static BIG_PHF: phf::Set<&str> = {};",
        big_set.build()
    )
    .unwrap();
}
