#![feature(test)]

#![feature(bench_black_box)]
extern crate test;

use std::hint::black_box;

use test::Bencher;

use simd256::SimdU256;

#[bench]
fn bench_mul(b: &mut Bencher) {
    let x = SimdU256::from(123456789123456789123456789);
    let y = SimdU256::from(123456789123456789123456789);
    b.iter(|| black_box(x * y));
}

#[bench]
fn bench_div(b: &mut Bencher) {
    let x = SimdU256::from(123456789123456789123456789);
    let y = SimdU256::from(123456789123456789123456789);
    b.iter(|| black_box(x / y));
}

#[bench]
fn bench_add(b: &mut Bencher) {
    let x = SimdU256::from(123456789123456789123456789);
    let y = SimdU256::from(123456789123456789123456789);
    b.iter(|| black_box(x + y));
}

#[bench]
fn bench_sub(b: &mut Bencher) {
    let x = SimdU256::from(123456789123456789123456789);
    let y = SimdU256::from(123456789123456789123456789);
    b.iter(|| black_box(x - y));
}
