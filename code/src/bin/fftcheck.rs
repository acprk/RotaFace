//! sanity: FFT owner re-encryption is correct (noise small) under the new key.
use rotaface::*;
fn main() {
    let p = params_from_env(); let mut rng = Rng::new(); let mut ctx = Ctx::new(&p);
    let s = keygen(&p, &mut rng); let s2 = keygen(&p, &mut rng);
    let emb = load_f32(&data_path(), 512);
    let x: Vec<i64> = { let mut v: Vec<i64> = emb[0].iter().map(|&f| (f * 500.0).round() as i64).collect(); v.resize(p.n, 0); v };
    let c = enc_template_b(&p, &s, &x, &mut rng);
    let (k1, k2) = (fkey(&p, &mut ctx, &s), fkey(&p, &mut ctx, &s2));
    let o = owner_reencrypt_fft(&p, &mut ctx, &k1, &k2, &c);
    println!("owner-fft re-encryption noise log2 std {:.2} (fresh {:.2})", std(&glwe_noise(&p, &s2, &o, &x)).log2(), std(&glwe_noise(&p, &s, &c, &x)).log2());
}
