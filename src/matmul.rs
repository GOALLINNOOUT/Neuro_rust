use std::sync::OnceLock;
#[cfg(test)]
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_storeu_pd,
};

pub(crate) fn avx2_available() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

#[cfg(test)]
static FORCE_LEGACY_SCALAR: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
pub(crate) fn set_force_legacy_scalar(force: bool) {
    FORCE_LEGACY_SCALAR.store(force, Ordering::Relaxed);
}

fn validate_matrices(a: &[Vec<f64>], b: &[Vec<f64>]) -> (usize, usize, usize) {
    let inner = a.first().map_or(0, Vec::len);
    let cols = b.first().map_or(0, Vec::len);
    assert!(a.iter().all(|row| row.len() == inner), "left matmul input must be rectangular");
    assert!(b.iter().all(|row| row.len() == cols), "right matmul input must be rectangular");
    assert_eq!(inner, b.len(), "matmul dimension mismatch");
    (a.len(), inner, cols)
}

fn matmul_scalar(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (rows, inner, cols) = validate_matrices(a, b);
    let mut out = vec![vec![0.0; cols]; rows];
    for i in 0..rows {
        for k in 0..inner {
            let aik = a[i][k];
            for j in 0..cols {
                out[i][j] += aik * b[k][j];
            }
        }
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_avx2(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let (rows, inner, cols) = validate_matrices(a, b);
    let mut out = vec![vec![0.0; cols]; rows];
    for i in 0..rows {
        for k in 0..inner {
            let a_broadcast = _mm256_set1_pd(a[i][k]);
            let mut j = 0;
            while j + 4 <= cols {
                // SAFETY: the loop bounds ensure four valid elements remain in both rows.
                let (b_values, accumulated) = unsafe {
                    (
                        _mm256_loadu_pd(b[k].as_ptr().add(j)),
                        _mm256_loadu_pd(out[i].as_ptr().add(j)),
                    )
                };
                let result = _mm256_fmadd_pd(a_broadcast, b_values, accumulated);
                // SAFETY: the loop bounds ensure four valid output elements remain.
                unsafe { _mm256_storeu_pd(out[i].as_mut_ptr().add(j), result) };
                j += 4;
            }
            for column in j..cols {
                out[i][column] += a[i][k] * b[k][column];
            }
        }
    }
    out
}

pub fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    validate_matrices(a, b);
    #[cfg(test)]
    if FORCE_LEGACY_SCALAR.load(Ordering::Relaxed) {
        return matmul_scalar(a, b);
    }
    if avx2_available() {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: avx2_available checks both target features before dispatch.
            unsafe { matmul_avx2(a, b) }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            matmul_scalar(a, b)
        }
    } else {
        matmul_scalar(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::{avx2_available, matmul, matmul_scalar};
    #[cfg(target_arch = "x86_64")]
    use super::matmul_avx2;
    use crate::attention::matrix_multiply;
    use std::hint::black_box;
    use std::time::Instant;

    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut value = self.0;
            value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            value ^ (value >> 31)
        }

        fn next_value(&mut self) -> f64 {
            let unit = (self.next_u64() >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64));
            unit * 4.0 - 2.0
        }
    }

    #[test]
    fn avx2_matches_scalar_on_random_matrices() {
        #[cfg(not(target_arch = "x86_64"))]
        {
            println!("AVX2 comparison skipped: target is not x86_64");
            return;
        }

        #[cfg(target_arch = "x86_64")]
        {
            if !avx2_available() {
                println!("AVX2 comparison skipped: AVX2/FMA unavailable");
                return;
            }

            for size in [8, 32, 128] {
                let mut rng = SplitMix64(0xA51C_2D7E_9134_B60F);
                let a: Vec<Vec<f64>> = (0..size)
                    .map(|_| (0..size).map(|_| rng.next_value()).collect())
                    .collect();
                let b: Vec<Vec<f64>> = (0..size)
                    .map(|_| (0..size).map(|_| rng.next_value()).collect())
                    .collect();
                let scalar = matmul_scalar(&a, &b);
                // SAFETY: the test checked runtime support immediately above.
                let simd = unsafe { super::matmul_avx2(&a, &b) };
                let max_delta = scalar.iter().flatten().zip(simd.iter().flatten())
                    .map(|(left, right)| (left - right).abs()).fold(0.0_f64, f64::max);
                println!("AVX2/scalar max delta for {size}x{size}: {max_delta:.3e}");
                assert!(max_delta < 1e-9, "{size}x{size} max delta was {max_delta}");

                let dispatched = matmul(&a, &b);
                let dispatch_delta = scalar.iter().flatten().zip(dispatched.iter().flatten())
                    .map(|(left, right)| (left - right).abs()).fold(0.0_f64, f64::max);
                assert!(dispatch_delta < 1e-9, "dispatch max delta was {dispatch_delta}");
            }
        }
    }

    #[test]
    #[ignore = "manual matmul microbenchmark"]
    fn matmul_bench() {
        const REPS: usize = 200;
        let shapes = [
            (15, 128, 32),
            (15, 32, 15),
            (15, 15, 32),
            (15, 128, 128),
            (15, 128, 512),
            (15, 512, 128),
        ];
        println!("variant,shape,repetitions,total_ms");
        for (rows, inner, cols) in shapes {
            let a: Vec<Vec<f64>> = (0..rows)
                .map(|i| (0..inner).map(|k| ((i * 13 + k * 7) % 31) as f64 / 31.0).collect())
                .collect();
            let b: Vec<Vec<f64>> = (0..inner)
                .map(|k| (0..cols).map(|j| ((k * 11 + j * 5) % 37) as f64 / 37.0).collect())
                .collect();
            let start = Instant::now();
            for _ in 0..REPS {
                black_box(matrix_multiply(black_box(&a), black_box(&b)).unwrap());
            }
            println!("legacy_scalar,{rows}x{inner}x{cols},{REPS},{:.3}", start.elapsed().as_secs_f64() * 1000.0);

            let start = Instant::now();
            for _ in 0..REPS {
                black_box(matmul(black_box(&a), black_box(&b)));
            }
            println!("direct_dispatch,{rows}x{inner}x{cols},{REPS},{:.3}", start.elapsed().as_secs_f64() * 1000.0);

            #[cfg(target_arch = "x86_64")]
            if avx2_available() {
                let start = Instant::now();
                for _ in 0..REPS {
                    // SAFETY: runtime detection above checked AVX2 and FMA support.
                    black_box(unsafe { matmul_avx2(black_box(&a), black_box(&b)) });
                }
                println!("avx2_kernel_direct,{rows}x{inner}x{cols},{REPS},{:.3}", start.elapsed().as_secs_f64() * 1000.0);
            }
        }
    }
}
