use std::collections::{HashMap, HashSet};

use divan::black_box;
use once_cell::sync::Lazy;
use radix_trie::Trie;
use trie_hard::TrieHard;

const OW_1984: &str = include_str!("../data/1984.txt");
const SUN_RISING: &str = include_str!("../data/sun-rising.txt");
const RANDOM: &str = include_str!("../data/random.txt");

// From https://github.com/pichillilorenzo/known-http-header-db/blob/main/src/db.json
const HEADERS: &str = include_str!("../data/headers.txt");
static HEADERS_REV: Lazy<Vec<String>> = Lazy::new(|| {
    HEADERS
        .lines()
        .collect::<HashSet<_>>()
        .into_iter()
        .map(|s| s.chars().rev().collect::<String>())
        .collect()
});

// Compile-time generated PHF Sets
include!(concat!(env!("OUT_DIR"), "/codegen.rs"));

const PERCENT: &[i32] = &[100, 75, 50, 25, 10, 5, 2, 1];

fn main() {
    divan::main();
}

/* -------------------------------------------------------------------------- */
/*                                 BENCHMARKS                                 */
/* -------------------------------------------------------------------------- */

#[divan::bench(args = args())]
fn trie_hard_get(bencher: divan::Bencher, input: &Input) {
    bencher
        .with_inputs(|| {
            let words = match input.size {
                Size::Header => get_header_text(),
                Size::Big => get_big_text(),
                Size::Small => get_small_text(),
            };
            let trie = make_trie(&words);
            (generate_samples(&words, input.percent), trie)
        })
        .bench_values(|(samples, trie): (Vec<&str>, TrieHard<'_, &str>)| {
            samples
                .iter()
                .filter_map(|w| trie.get(black_box(&w[..])))
                .count()
        });
}

#[divan::bench(args = args())]
fn radix_trie_get(bencher: divan::Bencher, input: &Input) {
    bencher
        .with_inputs(|| {
            let words = match input.size {
                Size::Header => get_header_text(),
                Size::Big => get_big_text(),
                Size::Small => get_small_text(),
            };
            let trie = make_radix_trie(&words);
            (generate_samples(&words, input.percent), trie)
        })
        .bench_values(|(samples, trie): (Vec<&str>, Trie<&str, usize>)| {
            samples
                .iter()
                .filter_map(|w| trie.get(black_box(&w[..])))
                .count()
        });
}

#[divan::bench(args = args())]
fn hashmap_get(bencher: divan::Bencher, input: &Input) {
    bencher
        .with_inputs(|| {
            let words = match input.size {
                Size::Header => get_header_text(),
                Size::Big => get_big_text(),
                Size::Small => get_small_text(),
            };
            let hashmap = make_hashmap(&words);
            (generate_samples(&words, input.percent), hashmap)
        })
        .bench_values(
            |(samples, hashmap): (Vec<&str>, HashMap<&str, &str>)| {
                samples
                    .iter()
                    .filter_map(|w| hashmap.get(black_box(&w[..])))
                    .count()
            },
        );
}

#[divan::bench(args = args())]
fn phf_get(bencher: divan::Bencher, input: &Input) {
    bencher
        .with_inputs(|| {
            let (words, phf) = match input.size {
                Size::Header => (get_header_text(), &HEADERS_PHF),
                Size::Big => (get_big_text(), &BIG_PHF),
                Size::Small => (get_small_text(), &SMALL_PHF),
            };
            (generate_samples(&words, input.percent), phf)
        })
        .bench_values(|(samples, phf): (Vec<&str>, &phf::Set<&str>)| {
            samples
                .iter()
                .filter_map(|w| phf.get_key(black_box(&w[..])))
                .count()
        });
}

#[divan::bench(args = &[Size::Big, Size::Small])]
fn trie_hard_insert(bencher: divan::Bencher, size: &Size) {
    bencher
        .with_inputs(|| match size {
            Size::Header => get_header_text(),
            Size::Big => get_big_text(),
            Size::Small => get_small_text(),
        })
        .bench_values(|words: Vec<&str>| make_trie(black_box(&words)));
}

#[divan::bench(args = &[Size::Big, Size::Small])]
fn radix_trie_insert(bencher: divan::Bencher, size: &Size) {
    bencher
        .with_inputs(|| match size {
            Size::Header => get_header_text(),
            Size::Big => get_big_text(),
            Size::Small => get_small_text(),
        })
        .bench_values(|words: Vec<&str>| make_radix_trie(black_box(&words)));
}

#[divan::bench(args = &[Size::Big, Size::Small])]
fn hashmap_insert(bencher: divan::Bencher, size: &Size) {
    bencher
        .with_inputs(|| match size {
            Size::Header => get_header_text(),
            Size::Big => get_big_text(),
            Size::Small => get_small_text(),
        })
        .bench_values(|words: Vec<&str>| make_hashmap(black_box(&words)));
}

/* -------------------------------------------------------------------------- */
/*                                   INPUTS                                   */
/* -------------------------------------------------------------------------- */

#[derive(Debug)]
enum Size {
    Header,
    Big,
    Small,
}

struct Input {
    size: Size,
    percent: i32,
}

impl std::fmt::Display for Input {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // divan sorts by lexicographic order, so we add padding to the percentage
        f.write_fmt(format_args!("{:?} - {:03}%", self.size, self.percent))
    }
}

fn args() -> impl Iterator<Item = Input> {
    PERCENT
        .iter()
        .map(|p| Input {
            size: Size::Header,
            percent: *p,
        })
        .chain(PERCENT.iter().map(|p| Input {
            size: Size::Big,
            percent: *p,
        }))
        .chain(PERCENT.iter().map(|p| Input {
            size: Size::Small,
            percent: *p,
        }))
}

/* -------------------------------------------------------------------------- */
/*                                   HELPERS                                  */
/* -------------------------------------------------------------------------- */

fn get_big_text() -> Vec<&'static str> {
    OW_1984
        .split(|c: char| c.is_whitespace())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect()
}

fn get_small_text() -> Vec<&'static str> {
    SUN_RISING
        .split(|c: char| c.is_whitespace())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect()
}

fn get_header_text() -> Vec<&'static str> {
    HEADERS_REV.iter().map(|s| s.as_str()).collect()
}

fn get_random_text() -> Vec<&'static str> {
    RANDOM
        .split(|c: char| c.is_whitespace())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect()
}

fn make_trie<'a>(words: &[&'a str]) -> TrieHard<'a, &'a str> {
    words.iter().copied().collect()
}

fn make_hashmap<'a>(words: &[&'a str]) -> HashMap<&'a str, &'a str> {
    words.iter().map(|k| (*k, *k)).collect()
}

fn make_radix_trie<'a>(words: &[&'a str]) -> Trie<&'a str, usize> {
    let mut trie = Trie::new();
    for w in words {
        trie.insert(&w[..], w.len());
    }
    trie
}

fn generate_samples<'a>(hits: &[&'a str], hit_percent: i32) -> Vec<&'a str> {
    let roulette_inc = hit_percent as f64 / 100.;
    let mut roulette = 0.;

    let mut result = get_random_text().to_owned();
    let mut hit_iter = hits.iter().cycle().copied();

    for w in result.iter_mut() {
        roulette += roulette_inc;
        if roulette >= 1. {
            roulette -= 1.;
            *w = hit_iter.next().unwrap();
        }
    }

    result
}
