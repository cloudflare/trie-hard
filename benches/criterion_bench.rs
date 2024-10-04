use std::collections::HashSet;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use once_cell::sync::Lazy;
use trie_hard::TrieHard;

/// This is a rip off of the benchmark suite for for
/// [`radix_trie`](https://github.com/michaelsproul/rust_radix_trie/blob/master/Cargo.toml)

const OW_1984: &str = include_str!("../data/1984.txt");
const SUN_RISING: &str = include_str!("../data/sun-rising.txt");
const RANDOM: &str = include_str!("../data/random.txt");

// From https://github.com/pichillilorenzo/known-http-header-db/blob/main/src/db.json
const HEADERS: &str = include_str!("../data/headers.txt");
static HEADERS_REV: Lazy<Vec<String>> = Lazy::new(|| {
    HEADERS
        .split(|c: char| c.is_whitespace())
        .collect::<HashSet<_>>()
        .into_iter()
        .map(|s| s.chars().rev().collect::<String>())
        .collect()
});

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

fn trie_insert_big(b: &mut Criterion) {
    let words = get_big_text();
    b.bench_function("trie hard insert - big", |b| {
        b.iter(|| make_trie(black_box(&words)))
    });
}

fn trie_insert_small(b: &mut Criterion) {
    let words = get_small_text();
    b.bench_function("trie hard insert - small", |b| {
        b.iter(|| make_trie(black_box(&words)))
    });
}

fn trie_insert_headers(b: &mut Criterion) {
    let words = get_header_text();
    b.bench_function("trie hard insert - headers", |b| {
        b.iter(|| make_trie(black_box(&words)))
    });
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

macro_rules! bench_percents_impl {
    ( [ $( ($size:expr, $percent:expr ), )+ ] ) => {$(
        paste::paste! {
            // Trie Hard
            fn [< trie_get_ $size _ $percent >] (b: &mut Criterion) {
                let words = [< get_ $size _text >]();
                let trie = make_trie(&words);
                let samples = generate_samples(&words, $percent);
                b.bench_function(
                    concat!(
                        "trie hard get - ",
                        stringify!($size),
                        " - ",
                        stringify!($percent),
                        "%"
                    ), |b| {
                    b.iter(|| {
                        samples.iter()
                            .filter_map(|w| trie.get(black_box(&w[..])))
                            .count()
                    })
                });
            }
        }


    )+};

    (  _groups [ $( ($size:expr, $percent:expr ), )+ ] ) => {
        paste::paste! {
            criterion_group!(
                get_benches,
                $(
                    [< trie_get_ $size _ $percent >],
                )+
            );
        }
    };
}

macro_rules! cartesian_impl {
    ($out:tt [] $b:tt $init_b:tt) => {
        bench_percents_impl!($out);
        bench_percents_impl!(_groups $out);
    };
    ($out:tt [$a:expr, $($at:tt)*] [] $init_b:tt) => {
        cartesian_impl!($out [$($at)*] $init_b $init_b);
    };
    ([$($out:tt)*] [$a:expr, $($at:tt)*] [$b:expr, $($bt:tt)*] $init_b:tt) => {
        cartesian_impl!([$($out)* ($a, $b),] [$a, $($at)*] [$($bt)*] $init_b);
    };
}

macro_rules! bench_get_percents {
    ([$($size:tt)*], [$($percent:tt)*]) => {
        cartesian_impl!([] [$($size)*,] [$($percent)*,] [$($percent)*,]);
    };
}

bench_get_percents!([header, big, small], [100, 75, 50, 25, 10, 5, 2, 1]);

criterion_group!(
    insert_benches,
    trie_insert_big,
    trie_insert_small,
    trie_insert_headers
);

criterion_main!(get_benches, insert_benches);
