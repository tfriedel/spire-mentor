//! CPython's `random.Random` (MT19937), bit for bit.

/// MT19937 as CPython drives it: `Random(seed)` → `init_by_array` over the
/// seed's 32-bit chunks; [`Self::random`] is `random.random()`,
/// [`Self::getrandbits`] the top `k` bits of one draw, [`Self::randbelow`]
/// rejection sampling on `n.bit_length()` bits. The one copy: the e2e
/// training env's F9 sampler and the workbench's deck analysis and SimLab
/// sweep both draw from it.
#[derive(Clone)]
pub struct PyRandom {
    mt: [u32; 624],
    index: usize,
}

impl std::fmt::Debug for PyRandom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyRandom")
            .field("index", &self.index)
            .finish_non_exhaustive()
    }
}

impl PyRandom {
    /// `random.Random(seed)` for a non-negative integer seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut key = Vec::new();
        let mut s = seed;
        loop {
            key.push(u32::try_from(s & 0xffff_ffff).expect("masked"));
            s >>= 32;
            if s == 0 {
                break;
            }
        }
        let mut r = Self {
            mt: [0; 624],
            index: 624,
        };
        r.init_by_array(&key);
        r
    }

    #[allow(clippy::cast_possible_truncation)]
    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..624 {
            self.mt[i] = 1_812_433_253_u32
                .wrapping_mul(self.mt[i - 1] ^ (self.mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        self.index = 624;
    }

    #[allow(clippy::cast_possible_truncation)]
    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19_650_218);
        let (mut i, mut j) = (1_usize, 0_usize);
        let n = 624;
        let mut k = n.max(key.len());
        while k > 0 {
            self.mt[i] = (self.mt[i]
                ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1_664_525)))
            .wrapping_add(key[j])
            .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= n {
                self.mt[0] = self.mt[n - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        k = n - 1;
        while k > 0 {
            self.mt[i] = (self.mt[i]
                ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1_566_083_941)))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= n {
                self.mt[0] = self.mt[n - 1];
                i = 1;
            }
            k -= 1;
        }
        self.mt[0] = 0x8000_0000;
    }

    fn generate(&mut self) {
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        const MATRIX_A: u32 = 0x9908_b0df;
        for i in 0..624 {
            let y = (self.mt[i] & UPPER) | (self.mt[(i + 1) % 624] & LOWER);
            let mut v = self.mt[(i + 397) % 624] ^ (y >> 1);
            if y & 1 != 0 {
                v ^= MATRIX_A;
            }
            self.mt[i] = v;
        }
        self.index = 0;
    }

    /// `genrand_uint32`.
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            self.generate();
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `getrandbits(k)` for `k <= 32`.
    pub fn getrandbits(&mut self, k: u32) -> u32 {
        if k == 0 {
            return 0;
        }
        self.next_u32() >> (32 - k)
    }

    /// `_randbelow_with_getrandbits(n)`.
    pub fn randbelow(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let k = 32 - n.leading_zeros();
        loop {
            let r = self.getrandbits(k);
            if r < n {
                return r;
            }
        }
    }

    /// `random.shuffle(x)`.
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            let j = self.randbelow(u32::try_from(i + 1).unwrap_or(u32::MAX)) as usize;
            x.swap(i, j);
        }
    }

    /// `random.random()`: 53 bits from two draws.
    pub fn random(&mut self) -> f64 {
        let a = f64::from(self.next_u32() >> 5);
        let b = f64::from(self.next_u32() >> 6);
        (a * 67_108_864.0 + b) * (1.0 / 9_007_199_254_740_992.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `random.Random(seed).random()` x3 in CPython, for a 32-bit seed and for
    /// an F9 seed (`(run seed << 32) ^ reseed`, two key words).
    #[test]
    fn py_random_matches_cpython() {
        let draws = |seed: u64| {
            let mut r = PyRandom::new(seed);
            [r.random(), r.random(), r.random()]
        };
        assert_eq!(
            draws(12345),
            [
                0.416_619_872_545_341_16,
                0.010_169_169_457_068_361,
                0.825_206_509_253_743_2
            ]
        );
        assert_eq!(
            draws((1_300_000_000 << 32) ^ 7),
            [
                0.578_388_400_415_640_4,
                0.044_378_682_949_188_9,
                0.923_261_666_672_245_7
            ]
        );
    }
}
