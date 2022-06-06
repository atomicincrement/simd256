#![feature(portable_simd)]

use std::simd::*;
use simd256::SimdU256;

#[test]
fn test_mul_special() {
    let a = SimdU256::from([0xffffffff; 8]);
    let b = SimdU256::from([0xffffffff; 8]);
    // println!("{:x}", a);
    // println!("{:x}", b);
    let c = a * b;
    // println!("{:x}", c);
    assert_eq!(c, SimdU256::from(1));
}

#[test]
fn test_div_special() {
    let a = SimdU256::from(1);
    let b = SimdU256::from(0);
    let c = a / b;
    assert_eq!(c, SimdU256::from([!0; 8]));

    let a = SimdU256::from(123456789123456789123456789);
    let b = SimdU256::from(123456789123456789123456789);
    let c = a / b;
    assert_eq!(c, SimdU256::from(1));

    let a = SimdU256::from(1);
    let b = SimdU256::from(1);
    let c = a / b;
    assert_eq!(c, SimdU256::from(1));

    let a = SimdU256::from(123456789123456789123456789);
    let b = SimdU256::from(1000000000);
    let c = a / b;
    assert_eq!(c, SimdU256::from(123456789123456789));

}

#[test]
fn test_mul_random() {
    use num_bigint::BigUint;
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mask = BigUint::from_slice(&[0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff]);
    let extra = BigUint::from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 1]);

    let to_simd = move |b: BigUint| {
        let mut r = (b & (&mask)).to_u32_digits();
        while r.len() < 8 {
            r.push(0);
        }
        r.reverse();
        SimdU256::from(<[u32; 8]>::try_from(&*r).unwrap())
    };

    for _ in 0..100000 {
        let lhs = [
            rng.gen(), rng.gen(), rng.gen(), rng.gen(),
            rng.gen(), rng.gen(), rng.gen(), rng.gen(),
        ];
        let rhs = [
            rng.gen(), rng.gen(), rng.gen(), rng.gen(),
            rng.gen(), rng.gen(), rng.gen(), rng.gen(),
        ];
        let mut lhsr = lhs;
        let mut rhsr = rhs;
        lhsr.reverse();
        rhsr.reverse();
        
        let lhs_uint = BigUint::from_slice(&lhsr);
        let rhs_uint = BigUint::from_slice(&rhsr);
        let lhs_simd = SimdU256::from(lhs);
        let rhs_simd = SimdU256::from(rhs);

        let prod_uint = to_simd(&lhs_uint * &rhs_uint);
        let prod_simd = lhs_simd * rhs_simd;
        assert_eq!(prod_uint, prod_simd);

        let prod_uint = to_simd(&lhs_uint * rhs[7]);
        let prod_simd = lhs_simd * rhs[7];
        assert_eq!(prod_uint, prod_simd);

        let sum_uint = to_simd(&lhs_uint + &rhs_uint);
        let sum_simd = lhs_simd + rhs_simd;
        assert_eq!(sum_uint, sum_simd);

        let diff_uint = to_simd(&extra + &lhs_uint - &rhs_uint);
        let diff_simd = lhs_simd - rhs_simd;
        assert_eq!(diff_uint, diff_simd);

        let sh = rng.gen::<u32>() & 255;
        let shl_uint = to_simd(&lhs_uint << sh);
        let shl_simd = lhs_simd << sh;
        assert_eq!(shl_uint, shl_simd);

        let shr_uint = to_simd(&lhs_uint >> sh);
        let shr_simd = lhs_simd >> sh;
        assert_eq!(shr_uint, shr_simd);

        assert_eq!((&lhs_uint >> sh).bits() as u32, shr_simd.bits());

        let div_uint = to_simd(&lhs_uint / (&rhs_uint >> 128));
        let div_simd = lhs_simd / (rhs_simd >> 128);
        // println!("uint={:x}", div_uint);
        // println!("simd={:x}", div_simd);
        assert_eq!(div_uint, div_simd);

    }
}
