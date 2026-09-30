//! RotaFace: server-side key rotation for FHE-encrypted biometric template databases.
//!
//! Two storage layouts are supported:
//!  * Layout A (GGSW-at-rest, BioVite-style): template = GGSW_s(x), query = GLWE_s(Δ·q~).
//!    Rotation: body rows are key-switched s -> s', mask rows are rebuilt from the new
//!    body rows by an external product with GGSW_{s'}(-s'_i) (RLWE->RGSW style).
//!  * Layout B (rotation-friendly): template = GLWE_s(Δ·x), query = GGSW_s(q~).
//!    Rotation: a single key switch s -> s'.
//! Key switching is realised as an external product with a "key-switching GGSW" whose
//! mask rows encrypt -β_j·s_i (old key) under s' and whose body row is trivial β_j.

use aligned_vec::ABox;
use concrete_fft::c64;
use tfhe::core_crypto::algorithms::polynomial_algorithms::polynomial_wrapping_mul;
use tfhe::core_crypto::algorithms::slice_algorithms::slice_wrapping_add_assign;
use tfhe::core_crypto::prelude::*;

pub type FGgsw = FourierGgswCiphertext<ABox<[c64]>>;
pub type Ggsw = GgswCiphertextOwned<u64>;
pub type Glwe = GlweCiphertextOwned<u64>;
pub type Sk = GlweSecretKeyOwned<u64>;

#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub k: usize,
    pub n: usize,
    /// gadget used by templates (layout A) / queries (layout B) at query time
    pub q_base_log: usize,
    pub q_level: usize,
    /// gadget used by the rotation token (KSG and RG_i)
    pub r_base_log: usize,
    pub r_level: usize,
    /// noise std relative to q = 2^64
    pub std_rel: f64,
    pub delta_log: usize,
    pub scale: f32,
}

impl Params {
    pub fn biovite() -> Self {
        Params { k: 3, n: 512, q_base_log: 16, q_level: 2, r_base_log: 4, r_level: 16,
                 std_rel: 3.2 / (1u64 << 40) as f64, delta_log: 45, scale: 500.0 }
    }
    pub fn glwe_size(&self) -> GlweSize { GlweSize(self.k + 1) }
    pub fn poly(&self) -> PolynomialSize { PolynomialSize(self.n) }
    pub fn delta(&self) -> u64 { 1u64 << self.delta_log }
    pub fn dist(&self) -> Gaussian<f64> { Gaussian::from_dispersion_parameter(StandardDev(self.std_rel), 0.0) }
    pub fn ggsw_bytes(&self, level: usize) -> usize { level * (self.k + 1) * (self.k + 1) * self.n * 8 }
    pub fn glwe_bytes(&self) -> usize { (self.k + 1) * self.n * 8 }
    pub fn fggsw_bytes(&self, level: usize) -> usize { level * (self.k + 1) * (self.k + 1) * (self.n / 2) * 16 }
}

pub fn modulus() -> CiphertextModulus<u64> { CiphertextModulus::new_native() }

pub struct Rng { pub secret: SecretRandomGenerator<ActivatedRandomGenerator>, pub enc: EncryptionRandomGenerator<ActivatedRandomGenerator> }
impl Rng {
    pub fn new() -> Self {
        let mut seeder = new_seeder();
        let secret = SecretRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed());
        let enc = EncryptionRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed(), seeder.as_mut());
        Rng { secret, enc }
    }
}

pub fn keygen(p: &Params, rng: &mut Rng) -> Sk {
    GlweSecretKey::generate_new_binary(GlweDimension(p.k), p.poly(), &mut rng.secret)
}

/// Per-thread FFT + scratch.
pub struct Ctx { pub fft: Fft, pub buf: ComputationBuffers, pub rr_buf: Option<ComputationBuffers> }
impl Ctx {
    pub fn new(p: &Params) -> Self {
        let fft = Fft::new(p.poly());
        let mut buf = ComputationBuffers::new();
        let req = add_external_product_assign_mem_optimized_requirement::<u64>(p.glwe_size(), p.poly(), fft.as_view())
            .unwrap().unaligned_bytes_required()
            .max(convert_standard_ggsw_ciphertext_to_fourier_mem_optimized_requirement(fft.as_view()).unwrap().unaligned_bytes_required());
        buf.resize(req);
        Ctx { fft, buf, rr_buf: None }
    }
    pub fn to_fourier(&mut self, g: &Ggsw) -> FGgsw {
        let mut f = FourierGgswCiphertext::new(g.glwe_size(), g.polynomial_size(), g.decomposition_base_log(), g.decomposition_level_count());
        convert_standard_ggsw_ciphertext_to_fourier_mem_optimized(g, &mut f, self.fft.as_view(), self.buf.stack());
        f
    }
    /// out += ggsw ⊡ glwe
    pub fn ext<O: ContainerMut<Element = u64>, I: Container<Element = u64>>(&mut self, out: &mut GlweCiphertext<O>, g: &FGgsw, ct: &GlweCiphertext<I>) {
        add_external_product_assign_mem_optimized(out, g, ct, self.fft.as_view(), self.buf.stack());
    }
}

// ---------------------------------------------------------------- polynomial helpers
pub fn pmul(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut out = Polynomial::from_container(vec![0u64; a.len()]);
    polynomial_wrapping_mul(&mut out, &Polynomial::from_container(a.to_vec()), &Polynomial::from_container(b.to_vec()));
    out.into_container()
}
pub fn neg(a: &[u64]) -> Vec<u64> { a.iter().map(|v| v.wrapping_neg()).collect() }
pub fn smul(a: &[u64], c: u64) -> Vec<u64> { a.iter().map(|v| v.wrapping_mul(c)).collect() }
pub fn key_poly(sk: &Sk, i: usize) -> Vec<u64> { sk.as_polynomial_list().get(i).as_ref().to_vec() }

/// Quantise an embedding to signed integers (wrapped into u64), padded to n.
pub fn quantise(p: &Params, f: &[f32]) -> Vec<i64> {
    let mut v: Vec<i64> = f.iter().map(|&x| (x * p.scale).round() as i64).collect();
    v.resize(p.n, 0);
    v
}
pub fn wrap(v: &[i64]) -> Vec<u64> { v.iter().map(|&x| x as u64).collect() }
/// Negacyclic twist so that coef_0(x · twist(q)) = <x, q>.
pub fn twist(q: &[i64]) -> Vec<i64> {
    let n = q.len();
    let mut t = vec![0i64; n];
    t[0] = q[0];
    for i in 1..n { t[n - i] = -q[i]; }
    t
}

// ---------------------------------------------------------------- GGSW with arbitrary row messages
/// Build a GGSW under `sk` whose row r at level m holds `msg(beta_m, r)` (added to a fresh
/// encryption of zero). If `trivial_body`, the body row (r == k) is noiseless.
pub fn ggsw_custom(p: &Params, sk: &Sk, base_log: usize, level: usize, rng: &mut Rng,
                   trivial_body: bool, msg: &dyn Fn(u64, usize) -> Vec<u64>) -> Ggsw {
    let mut g = GgswCiphertext::new(0u64, p.glwe_size(), p.poly(), DecompositionBaseLog(base_log), DecompositionLevelCount(level), modulus());
    encrypt_constant_ggsw_ciphertext(sk, &mut g, Plaintext(0), p.dist(), &mut rng.enc);
    for (m, mut lm) in g.iter_mut().enumerate() {
        let beta = 1u64 << (64 - base_log * (m + 1));
        for (r, mut row) in lm.as_mut_glwe_list().iter_mut().enumerate() {
            if trivial_body && r == p.k { row.as_mut().fill(0); }
            let v = msg(beta, r);
            slice_wrapping_add_assign(row.get_mut_body().as_mut(), &v);
        }
    }
    g
}

/// Standard GGSW_sk(m) for a polynomial message m.
pub fn ggsw_encrypt_poly(p: &Params, sk: &Sk, m: &[u64], base_log: usize, level: usize, rng: &mut Rng) -> Ggsw {
    let s: Vec<Vec<u64>> = (0..p.k).map(|i| key_poly(sk, i)).collect();
    let k = p.k;
    ggsw_custom(p, sk, base_log, level, rng, false, &|beta, r| {
        let bm = smul(m, beta);
        if r < k { neg(&pmul(&bm, &s[r])) } else { bm }
    })
}

pub fn glwe_encrypt(p: &Params, sk: &Sk, m: &[u64], rng: &mut Rng) -> Glwe {
    let mut ct = GlweCiphertext::new(0u64, p.glwe_size(), p.poly(), modulus());
    encrypt_glwe_ciphertext(sk, &mut ct, &PlaintextList::from_container(m.to_vec()), p.dist(), &mut rng.enc);
    ct
}

pub fn glwe_phase<C: Container<Element = u64>>(p: &Params, sk: &Sk, ct: &GlweCiphertext<C>) -> Vec<u64> {
    let mut pt = PlaintextList::new(0u64, PlaintextCount(p.n));
    decrypt_glwe_ciphertext(sk, ct, &mut pt);
    pt.into_container()
}

// ---------------------------------------------------------------- rotation token
pub struct Token { pub ksg: FGgsw, pub rg: Vec<FGgsw> }
pub struct TokenStd { pub ksg: Ggsw, pub rg: Vec<Ggsw> }

impl TokenStd {
    pub fn bytes(&self, p: &Params) -> usize { (1 + self.rg.len()) * p.ggsw_bytes(p.r_level) }
}

/// Token for s -> s2. With `with_rg == false` only the key-switching part (layout B).
pub fn token_gen(p: &Params, s: &Sk, s2: &Sk, with_rg: bool, rng: &mut Rng) -> TokenStd {
    let k = p.k;
    let so: Vec<Vec<u64>> = (0..k).map(|i| key_poly(s, i)).collect();
    let sn: Vec<Vec<u64>> = (0..k).map(|i| key_poly(s2, i)).collect();
    let n = p.n;
    let ksg = ggsw_custom(p, s2, p.r_base_log, p.r_level, rng, true, &|beta, r| {
        if r < k { neg(&smul(&so[r], beta)) } else { let mut v = vec![0u64; n]; v[0] = beta; v }
    });
    let mut rg = Vec::new();
    if with_rg {
        for i in 0..k {
            let y = neg(&sn[i]); // message -s'_i
            rg.push(ggsw_encrypt_poly(p, s2, &y, p.r_base_log, p.r_level, rng));
        }
    }
    TokenStd { ksg, rg }
}

impl Token {
    pub fn from_std(ctx: &mut Ctx, t: &TokenStd) -> Self {
        Token { ksg: ctx.to_fourier(&t.ksg), rg: t.rg.iter().map(|g| ctx.to_fourier(g)).collect() }
    }
}

// ---------------------------------------------------------------- rotation
/// Layout B: GLWE_s(m) -> GLWE_s'(m).
pub fn rotate_glwe(ctx: &mut Ctx, tok: &Token, ct: &Glwe) -> Glwe {
    let mut out = GlweCiphertext::new(0u64, ct.glwe_size(), ct.polynomial_size(), modulus());
    ctx.ext(&mut out, &tok.ksg, ct);
    out
}

/// Layout A: GGSW_s(x) -> GGSW_s'(x).
pub fn rotate_ggsw(p: &Params, ctx: &mut Ctx, tok: &Token, g: &Ggsw) -> Ggsw {
    let k = p.k;
    let mut out = GgswCiphertext::new(0u64, g.glwe_size(), g.polynomial_size(), g.decomposition_base_log(), g.decomposition_level_count(), modulus());
    for (lm_in, mut lm_out) in g.iter().zip(out.iter_mut()) {
        let rows_in = lm_in.as_glwe_list();
        let body_in = rows_in.get(k);
        let mut body = GlweCiphertext::new(0u64, p.glwe_size(), p.poly(), modulus());
        ctx.ext(&mut body, &tok.ksg, &body_in);
        let mut rows_out = lm_out.as_mut_glwe_list();
        for r in 0..k {
            let mut row = rows_out.get_mut(r);
            ctx.ext(&mut row, &tok.rg[r], &body);
        }
        rows_out.get_mut(k).as_mut().copy_from_slice(body.as_ref());
    }
    out
}

// ---------------------------------------------------------------- scoring
/// Layout A score: template GGSW (Fourier) ⊡ query GLWE(Δ·twist(q)); returns phase(coef0)/Δ.
pub fn score_a(p: &Params, ctx: &mut Ctx, sk: &Sk, tmpl: &FGgsw, query: &Glwe) -> f64 {
    let mut out = GlweCiphertext::new(0u64, p.glwe_size(), p.poly(), modulus());
    ctx.ext(&mut out, tmpl, query);
    (glwe_phase(p, sk, &out)[0] as i64) as f64 / p.delta() as f64
}
/// Layout B score: query GGSW(twist(q)) ⊡ template GLWE(Δ·x).
pub fn score_b(p: &Params, ctx: &mut Ctx, sk: &Sk, query: &FGgsw, tmpl: &Glwe) -> f64 {
    let mut out = GlweCiphertext::new(0u64, p.glwe_size(), p.poly(), modulus());
    ctx.ext(&mut out, query, tmpl);
    (glwe_phase(p, sk, &out)[0] as i64) as f64 / p.delta() as f64
}
/// Server-side only (no decryption): the encrypted score GLWE, layout A.
pub fn eval_a(p: &Params, ctx: &mut Ctx, tmpl: &FGgsw, query: &Glwe) -> Glwe {
    let mut out = GlweCiphertext::new(0u64, p.glwe_size(), p.poly(), modulus());
    ctx.ext(&mut out, tmpl, query);
    out
}

// ---------------------------------------------------------------- encoders for the two layouts
pub fn enc_template_a(p: &Params, sk: &Sk, x: &[i64], rng: &mut Rng) -> Ggsw { ggsw_encrypt_poly(p, sk, &wrap(x), p.q_base_log, p.q_level, rng) }
pub fn enc_query_a(p: &Params, sk: &Sk, q: &[i64], rng: &mut Rng) -> Glwe { glwe_encrypt(p, sk, &smul(&wrap(&twist(q)), p.delta()), rng) }
pub fn enc_template_b(p: &Params, sk: &Sk, x: &[i64], rng: &mut Rng) -> Glwe { glwe_encrypt(p, sk, &smul(&wrap(x), p.delta()), rng) }
pub fn enc_query_b(p: &Params, sk: &Sk, q: &[i64], rng: &mut Rng) -> Ggsw { ggsw_encrypt_poly(p, sk, &wrap(&twist(q)), p.q_base_log, p.q_level, rng) }

/// Key-owner baseline: recover x from a layout-A template (decrypt the level-1 body row).
pub fn dec_template_a(p: &Params, sk: &Sk, g: &Ggsw) -> Vec<i64> {
    let lm = g.iter().next().unwrap();
    let rows = lm.as_glwe_list();
    let ph = glwe_phase(p, sk, &rows.get(p.k));
    let sh = 64 - p.q_base_log;
    ph.iter().map(|&v| ((v as i64 as f64) / (1u64 << sh) as f64).round() as i64).collect()
}
pub fn dec_template_b(p: &Params, sk: &Sk, ct: &Glwe) -> Vec<i64> {
    glwe_phase(p, sk, ct).iter().map(|&v| ((v as i64) as f64 / p.delta() as f64).round() as i64).collect()
}

// ---------------------------------------------------------------- noise measurement
fn diff_std(ph: &[u64], exp: &[u64], acc: &mut Vec<f64>) {
    for (a, b) in ph.iter().zip(exp) { acc.push(a.wrapping_sub(*b) as i64 as f64); }
}
/// Row-noise samples of a layout-A template under key sk: (body-row errors, mask-row errors).
pub fn ggsw_row_noise(p: &Params, sk: &Sk, g: &Ggsw, x: &[i64]) -> (Vec<f64>, Vec<f64>) {
    let k = p.k;
    let s: Vec<Vec<u64>> = (0..k).map(|i| key_poly(sk, i)).collect();
    let xm = wrap(x);
    let (mut body, mut mask) = (Vec::new(), Vec::new());
    for (m, lm) in g.iter().enumerate() {
        let beta = 1u64 << (64 - g.decomposition_base_log().0 * (m + 1));
        let bm = smul(&xm, beta);
        for (r, row) in lm.as_glwe_list().iter().enumerate() {
            let ph = glwe_phase(p, sk, &row);
            if r < k { diff_std(&ph, &neg(&pmul(&bm, &s[r])), &mut mask); } else { diff_std(&ph, &bm, &mut body); }
        }
    }
    (body, mask)
}
pub fn glwe_noise(p: &Params, sk: &Sk, ct: &Glwe, x: &[i64]) -> Vec<f64> {
    let mut v = Vec::new();
    diff_std(&glwe_phase(p, sk, ct), &smul(&wrap(x), p.delta()), &mut v);
    v
}
pub fn std(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n).sqrt()
}
pub fn exact_ip(x: &[i64], q: &[i64]) -> i64 { x.iter().zip(q).map(|(a, b)| a * b).sum() }

// ---------------------------------------------------------------- data
pub fn load_f32(path: &str, dim: usize) -> Vec<Vec<f32>> {
    let b = std::fs::read(path).expect("read embeddings");
    let f: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    f.chunks_exact(dim).map(|c| c.to_vec()).collect()
}
pub fn data_path() -> String {
    std::env::var("RF_DATA").unwrap_or_else(|_| "data/lfw_emb.f32".into())
}

/// Fast decryption of coefficient 0 only (what the protocol needs): phase_0 / Δ.
pub fn coef0_score<C: Container<Element = u64>>(p: &Params, sk_polys: &[Vec<u64>], ct: &GlweCiphertext<C>) -> f64 {
    let n = p.n;
    let data = ct.as_ref();
    let mut ph = data[p.k * n]; // b_0
    for i in 0..p.k {
        let a = &data[i * n..(i + 1) * n];
        let s = &sk_polys[i];
        // coef0(a*s) = a0*s0 - sum_{j>=1} a_j * s_{n-j}
        let mut acc = a[0].wrapping_mul(s[0]);
        for j in 1..n { acc = acc.wrapping_sub(a[j].wrapping_mul(s[n - j])); }
        ph = ph.wrapping_sub(acc);
    }
    (ph as i64) as f64 / p.delta() as f64
}
pub fn key_polys(p: &Params, sk: &Sk) -> Vec<Vec<u64>> { (0..p.k).map(|i| key_poly(sk, i)).collect() }

impl Params {
    /// Main 128-bit parameter set: n = k*N = 2048 (lattice-estimator rough: 2^140.7).
    pub fn p128() -> Self {
        Params { k: 1, n: 2048, q_base_log: 16, q_level: 2, r_base_log: 4, r_level: 15,
                 std_rel: 3.2 / (1u64 << 40) as f64, delta_log: 45, scale: 500.0 }
    }
    pub fn by_name(name: &str) -> Self { match name { "p128" => Self::p128(), _ => Self::biovite() } }
    /// number of 512-dim templates packed per polynomial
    pub fn slots(&self, d: usize) -> usize { self.n / d }
}
/// Pack several quantised templates (each of length d) at offsets j*d.
pub fn pack(p: &Params, xs: &[Vec<i64>], d: usize) -> Vec<i64> {
    let mut v = vec![0i64; p.n];
    for (j, x) in xs.iter().enumerate() { for i in 0..d { v[j * d + i] = x[i]; } }
    v
}
/// Decrypt coefficient c only: phase_c / Δ.
pub fn coef_score<C: Container<Element = u64>>(p: &Params, sk_polys: &[Vec<u64>], ct: &GlweCiphertext<C>, c: usize) -> f64 {
    let n = p.n;
    let data = ct.as_ref();
    let mut ph = data[p.k * n + c];
    for i in 0..p.k {
        let a = &data[i * n..(i + 1) * n];
        let s = &sk_polys[i];
        let mut acc = 0u64;
        for j in 0..=c { acc = acc.wrapping_add(a[j].wrapping_mul(s[c - j])); }
        for j in (c + 1)..n { acc = acc.wrapping_sub(a[j].wrapping_mul(s[n + c - j])); }
        ph = ph.wrapping_sub(acc);
    }
    (ph as i64) as f64 / p.delta() as f64
}
pub fn params_from_env() -> Params {
    let mut p = Params::by_name(&std::env::var("RF_PARAMS").unwrap_or_else(|_| "biovite".into()));
    if let Ok(v) = std::env::var("RF_K") { p.k = v.parse().unwrap(); }
    if let Ok(v) = std::env::var("RF_N") { p.n = v.parse().unwrap(); }
    if let Ok(v) = std::env::var("RF_LOGSIGMA") { p.std_rel = 2f64.powf(v.parse::<f64>().unwrap() - 64.0); }
    p
}

// ---------------------------------------------------------------- hardness upgrade (k -> k2, same N)
impl Params {
    /// Upgrade target for the BioVite shape: k=4, N=512 (n = 2048, same ring, same q, same sigma).
    pub fn p128k4() -> Self { let mut p = Self::biovite(); p.k = 4; p }
}
/// Pad a GLWE under (s_1..s_k) to a GLWE under (s_1..s_k, 0, ..) of GLWE dimension k2.
pub fn pad_glwe<C: Container<Element = u64>>(p: &Params, k2: usize, ct: &GlweCiphertext<C>) -> Glwe {
    let n = p.n; let d = ct.as_ref();
    let mut v = vec![0u64; (k2 + 1) * n];
    v[..p.k * n].copy_from_slice(&d[..p.k * n]);
    v[k2 * n..].copy_from_slice(&d[p.k * n..]);
    GlweCiphertext::from_container(v, PolynomialSize(n), modulus())
}
/// Old key padded with zero polynomials to dimension k2.
pub fn pad_key(p: &Params, k2: usize, sk: &Sk) -> Sk {
    let mut v = sk.as_ref().to_vec(); v.resize(k2 * p.n, 0);
    GlweSecretKey::from_container(v, PolynomialSize(p.n))
}
/// Layout B upgrade: GLWE_s(m) (dim k) -> GLWE_s2(m) (dim k2), token built with p2 and pad_key(s).
pub fn upgrade_glwe(p: &Params, p2: &Params, ctx2: &mut Ctx, tok: &Token, ct: &Glwe) -> Glwe {
    rotate_glwe(ctx2, tok, &pad_glwe(p, p2.k, ct))
}
/// Layout A upgrade: rebuild a GGSW of dimension k2 from the (padded, key-switched) body rows.
pub fn upgrade_ggsw(p: &Params, p2: &Params, ctx2: &mut Ctx, tok: &Token, g: &Ggsw) -> Ggsw {
    let k2 = p2.k;
    let mut out = GgswCiphertext::new(0u64, p2.glwe_size(), p2.poly(), g.decomposition_base_log(), g.decomposition_level_count(), modulus());
    for (lm_in, mut lm_out) in g.iter().zip(out.iter_mut()) {
        let rows_in = lm_in.as_glwe_list();
        let body_in = pad_glwe(p, k2, &rows_in.get(p.k));
        let mut body = GlweCiphertext::new(0u64, p2.glwe_size(), p2.poly(), modulus());
        ctx2.ext(&mut body, &tok.ksg, &body_in);
        let mut rows_out = lm_out.as_mut_glwe_list();
        for r in 0..k2 { let mut row = rows_out.get_mut(r); ctx2.ext(&mut row, &tok.rg[r], &body); }
        rows_out.get_mut(k2).as_mut().copy_from_slice(body.as_ref());
    }
    out
}

// ---------------------------------------------------------------- re-randomisation (public key under s')
use tfhe::core_crypto::fft_impl::fft64::math::polynomial::FourierPolynomial;
use rand::Rng as _;

/// RLWE public key pk = GLWE_{s'}(0) = (a_1..a_k, b = sum a_i s_i + e), stored in the Fourier domain.
pub struct PubKey { pub mask: Vec<FourierPolynomial<ABox<[c64]>>>, pub body: FourierPolynomial<ABox<[c64]>>, pub sigma_abs: f64, pub tuniform_bits: u32 }

pub fn pk_gen(p: &Params, ctx: &mut Ctx, sk: &Sk, rng: &mut Rng) -> PubKey {
    let z = glwe_encrypt(p, sk, &vec![0u64; p.n], rng);
    let fft = ctx.fft.as_view();
    let req = fft.forward_scratch().unwrap().or(fft.backward_scratch().unwrap()).unaligned_bytes_required();
    let mut buf = ComputationBuffers::new(); buf.resize(req);
    let d = z.as_ref();
    let conv = |sl: &[u64], buf: &mut ComputationBuffers| {
        let mut f = FourierPolynomial::new(p.poly());
        fft.forward_as_torus(f.as_mut_view(), Polynomial::from_container(sl).as_view(), buf.stack());
        f
    };
    let mask = (0..p.k).map(|i| conv(&d[i * p.n..(i + 1) * p.n], &mut buf)).collect();
    let body = conv(&d[p.k * p.n..], &mut buf);
    PubKey { mask, body, sigma_abs: p.std_rel * 2f64.powi(64), tuniform_bits: 26 }
}

#[allow(dead_code)]
fn gauss(r: &mut impl rand::Rng, s: f64) -> u64 {
    // Box-Muller
    let (u1, u2): (f64, f64) = (r.gen::<f64>().max(1e-300), r.gen());
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() * s).round() as i64 as u64
}

/// ct += (a*u + e1, b*u + e2): a fresh encryption of zero under s' with server randomness.
pub fn rerandomise(p: &Params, ctx: &mut Ctx, pk: &PubKey, ct: &mut Glwe) {
    let mut r = rand::thread_rng();
    let n = p.n;
    let u: Vec<u64> = (0..n).map(|_| r.gen::<bool>() as u64).collect();
    let fft = ctx.fft.as_view();
    let req = fft.forward_scratch().unwrap().or(fft.backward_scratch().unwrap()).unaligned_bytes_required();
    if ctx.rr_buf.is_none() { let mut b = ComputationBuffers::new(); b.resize(req); ctx.rr_buf = Some(b); }
    let buf = ctx.rr_buf.as_mut().unwrap();
    let mut fu = FourierPolynomial::new(p.poly());
    fft.forward_as_integer(fu.as_mut_view(), Polynomial::from_container(u.as_slice()).as_view(), buf.stack());
    let mut tmp = FourierPolynomial::new(p.poly());
    let d = ct.as_mut();
    for i in 0..=p.k {
        let src = if i < p.k { &pk.mask[i] } else { &pk.body };
        for ((t, a), b) in tmp.data.iter_mut().zip(src.data.iter()).zip(fu.data.iter()) { *t = *a * *b; }
        let slice = &mut d[i * n..(i + 1) * n];
        fft.add_backward_as_torus(Polynomial::from_container(&mut *slice).as_mut_view(), tmp.as_view(), buf.stack());
        // TUniform(b) noise (as adopted by tfhe-rs): uniform integer in [-2^b, 2^b], std ~ 2^b/sqrt(3)
        let b = pk.tuniform_bits;
        for v in slice.iter_mut() { let e = (r.gen::<u64>() % ((2u64 << b) + 1)) as i64 - (1i64 << b); *v = v.wrapping_add(e as u64); }
    }
}

// ---------------------------------------------------------------- noise-adaptive template compression
/// Round every coefficient of a GLWE to the nearest multiple of 2^drop (keep 64-drop high bits).
pub fn truncate_glwe(ct: &mut Glwe, drop: u32) {
    if drop == 0 { return; }
    let half = 1u64 << (drop - 1);
    let mask = !((1u64 << drop) - 1);
    for v in ct.as_mut().iter_mut() { *v = v.wrapping_add(half) & mask; }
}
/// Largest drop such that rounding noise std  sqrt(1 + k*N/2) * 2^drop / sqrt(12) <= frac * sigma_T.
pub fn adaptive_drop(p: &Params, sigma_t: f64, frac: f64) -> u32 {
    let amp = (1.0 + (p.k * p.n) as f64 / 2.0).sqrt() / 12f64.sqrt();
    let d = (frac * sigma_t / amp).log2().floor();
    d.max(0.0).min(40.0) as u32
}
/// Model template noise after t rotations (no truncation): sigma_0^2 + t * k N l B^2/12 sigma^2
pub fn model_sigma_t(p: &Params, t: usize) -> f64 {
    let s = p.std_rel * 2f64.powi(64);
    let step = (p.k * p.n * p.r_level) as f64 * 2f64.powi(2 * p.r_base_log as i32) / 12.0 * s * s;
    (s * s + t as f64 * step).sqrt()
}

// ---------------------------------------------------------------- compact pipeline: seeded native key switching
/// Transported token: seeded GLWE list (only bodies + 16-byte seed). Row (l*k + i) encrypts -beta_l * s_i under s'.
pub struct SeededKsk { pub list: SeededGlweCiphertextList<Vec<u64>>, pub bl: usize, pub lv: usize }
impl SeededKsk { pub fn bytes(&self) -> usize { self.list.as_ref().len() * 8 + 16 } }

pub fn ksk_gen_seeded(p: &Params, s_old: &Sk, s_new: &Sk, bl: usize, lv: usize) -> SeededKsk {
    let (k, n) = (p.k, p.n);
    let so: Vec<Vec<u64>> = (0..k).map(|i| key_poly(s_old, i)).collect();
    let mut pt = Vec::with_capacity(lv * k * n);
    for l in 0..lv { let beta = 1u64 << (64 - bl * (l + 1)); for i in 0..k { pt.extend(neg(&smul(&so[i], beta))); } }
    let mut seeder = new_seeder();
    let mut list = SeededGlweCiphertextList::new(0u64, p.glwe_size(), p.poly(), GlweCiphertextCount(lv * k), tfhe::core_crypto::commons::math::random::CompressionSeed { seed: seeder.seed() }, modulus());
    encrypt_seeded_glwe_ciphertext_list(s_new, &mut list, &PlaintextList::from_container(pt), p.dist(), seeder.as_mut());
    SeededKsk { list, bl, lv }
}

type FPoly = FourierPolynomial<ABox<[c64]>>;
/// Server-side expanded key-switching key in the Fourier domain: rows[l*k + i][poly].
pub struct NativeKsk { pub bl: usize, pub lv: usize, pub rows: Vec<Vec<FPoly>> }
impl NativeKsk {
    pub fn expand(p: &Params, ctx: &mut Ctx, s: &SeededKsk) -> Self {
        let mut out = GlweCiphertextList::new(0u64, p.glwe_size(), p.poly(), GlweCiphertextCount(s.lv * p.k), modulus());
        decompress_seeded_glwe_ciphertext_list::<_, _, _, ActivatedRandomGenerator>(&mut out, &s.list);
        let rows = out.iter().map(|row| row.as_polynomial_list().iter().map(|poly| {
            let mut f = FPoly::new(p.poly());
            ctx.fft.as_view().forward_as_torus(f.as_mut_view(), poly, ctx.buf.stack()); f }).collect()).collect();
        NativeKsk { bl: s.bl, lv: s.lv, rows }
    }
}

/// Native GLWE key switch: decompose only the mask (top bl*lv bits, signed digits), body copied.
pub fn native_ks(p: &Params, ctx: &mut Ctx, t: &NativeKsk, ct: &Glwe) -> Glwe {
    let (k, n) = (p.k, p.n);
    let d = ct.as_ref();
    let mut acc: Vec<Vec<c64>> = (0..=k).map(|_| vec![c64::new(0.0, 0.0); n / 2]).collect();
    let mut dig = vec![vec![0u64; n]; t.lv];
    let mut xs = vec![0u64; n];
    let mut f = FPoly::new(p.poly());
    let tot = t.bl * t.lv; let bmask = (1u64 << t.bl) - 1; let half = 1u64 << (t.bl - 1);
    for i in 0..k {
        let a = &d[i * n..(i + 1) * n];
        let sh = 64 - tot; let rnd = 1u64 << (63 - tot);
        for c in 0..n { xs[c] = a[c].wrapping_add(rnd) >> sh; }
        for l in (0..t.lv).rev() {
            for (dc, xc) in dig[l].iter_mut().zip(xs.iter_mut()) {
                let x = *xc; let di = x & bmask; let carry = (di >= half) as u64;
                *dc = di.wrapping_sub(carry << t.bl); *xc = (x >> t.bl).wrapping_add(carry);
            }
        }
        for l in 0..t.lv {
            ctx.fft.as_view().forward_as_integer(f.as_mut_view(), Polynomial::from_container(dig[l].as_slice()), ctx.buf.stack());
            let row = &t.rows[l * k + i];
            for pp in 0..=k {
                for ((a_, u), v) in acc[pp].iter_mut().zip(f.data.iter()).zip(row[pp].data.iter()) { *a_ += *u * *v; }
            }
        }
    }
    let mut out = GlweCiphertext::new(0u64, ct.glwe_size(), ct.polynomial_size(), modulus());
    out.get_mut_body().as_mut().copy_from_slice(&d[k * n..]);
    {
        let mut pl = out.as_mut_polynomial_list();
        for pp in 0..=k {
            let fa = FourierPolynomial { data: acc[pp].as_slice() };
            ctx.fft.as_view().add_backward_as_torus(pl.get_mut(pp), fa, ctx.buf.stack());
        }
    }
    out
}

/// Round mask coefficients to keep `mask_bits` high bits and body coefficients to keep `body_bits`.
pub fn truncate_split(p: &Params, ct: &mut Glwe, mask_bits: u32, body_bits: u32) {
    let n = p.n; let k = p.k;
    let rd = |v: &mut u64, keep: u32| { if keep >= 64 { return; } let dr = 64 - keep; let half = 1u64 << (dr - 1); *v = v.wrapping_add(half) & !((1u64 << dr) - 1); };
    let d = ct.as_mut();
    for v in d[..k * n].iter_mut() { rd(v, mask_bits); }
    for v in d[k * n..].iter_mut() { rd(v, body_bits); }
}
pub fn stored_bytes_per_template(p: &Params, mask_bits: u32, body_bits: u32) -> f64 {
    (p.k * p.n * mask_bits as usize + p.n * body_bits as usize) as f64 / 8.0 / p.slots(512) as f64
}

// ---------------------------------------------------------------- FFT-based owner baseline (fair comparison)
/// Precomputed Fourier transform of the secret key polynomials (as integers).
pub struct FKey { pub f: Vec<FPoly> }
pub fn fkey(p: &Params, ctx: &mut Ctx, sk: &Sk) -> FKey {
    let fft = ctx.fft.as_view();
    let req = fft.forward_scratch().unwrap().or(fft.backward_scratch().unwrap()).unaligned_bytes_required();
    let mut buf = ComputationBuffers::new(); buf.resize(req);
    FKey { f: (0..p.k).map(|i| { let kp = key_poly(sk, i); let mut f = FPoly::new(p.poly());
        fft.forward_as_integer(f.as_mut_view(), Polynomial::from_container(kp.as_slice()), buf.stack()); f }).collect() }
}
fn ensure_rr(ctx: &mut Ctx) {
    if ctx.rr_buf.is_none() {
        let fft = ctx.fft.as_view();
        let req = fft.forward_scratch().unwrap().or(fft.backward_scratch().unwrap()).unaligned_bytes_required();
        let mut b = ComputationBuffers::new(); b.resize(req); ctx.rr_buf = Some(b);
    }
}
/// <a, s> via FFT, returned in the standard domain.
fn inner_as(p: &Params, ctx: &mut Ctx, k: &FKey, mask: &[u64]) -> Vec<u64> {
    ensure_rr(ctx);
    let n = p.n; let fft = ctx.fft.as_view();
    let buf = ctx.rr_buf.as_mut().unwrap();
    let mut acc = vec![c64::new(0.0, 0.0); n / 2];
    let mut fa = FPoly::new(p.poly());
    for i in 0..p.k {
        fft.forward_as_torus(fa.as_mut_view(), Polynomial::from_container(&mask[i * n..(i + 1) * n]).as_view(), buf.stack());
        for ((z, x), y) in acc.iter_mut().zip(fa.data.iter()).zip(k.f[i].data.iter()) { *z += *x * *y; }
    }
    let mut out = vec![0u64; n];
    fft.add_backward_as_torus(Polynomial::from_container(out.as_mut_slice()).as_mut_view(), FourierPolynomial { data: acc.as_slice() }, buf.stack());
    out
}
/// Owner path per ciphertext: decrypt under s (FFT), re-encrypt under s2 (FFT, uniform mask, TUniform(2^26) noise).
pub fn owner_reencrypt_fft(p: &Params, ctx: &mut Ctx, k_old: &FKey, k_new: &FKey, ct: &Glwe) -> Glwe {
    let n = p.n; let d = ct.as_ref();
    let as_old = inner_as(p, ctx, k_old, &d[..p.k * n]);
    let m: Vec<u64> = d[p.k * n..].iter().zip(&as_old).map(|(b, a)| b.wrapping_sub(*a)).collect();
    let mut r = rand::thread_rng();
    let mut out = vec![0u64; (p.k + 1) * n];
    for v in out[..p.k * n].iter_mut() { *v = r.gen(); }
    let as_new = inner_as(p, ctx, k_new, &out[..p.k * n]);
    for c in 0..n { let e = (r.gen::<u64>() % ((2u64 << 26) + 1)) as i64 - (1i64 << 26); out[p.k * n + c] = as_new[c].wrapping_add(m[c]).wrapping_add(e as u64); }
    GlweCiphertext::from_container(out, p.poly(), modulus())
}
