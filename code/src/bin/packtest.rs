//! Check slot packing at p128: 4 templates per GLWE/GGSW, scores at coefficients j*512.
use rotaface::*;
fn main() {
    let p = Params::p128();
    let d = 512; let sl = p.slots(d);
    let emb: Vec<Vec<f32>> = load_f32(&data_path(), 512).into_iter().filter(|v| v.iter().map(|x| x * x).sum::<f32>() > 0.25).collect();
    let q512 = |v: &Vec<f32>| -> Vec<i64> { v.iter().map(|&x| (x * p.scale).round() as i64).collect() };
    let mut rng = Rng::new(); let mut ctx = Ctx::new(&p);
    let mut s = keygen(&p, &mut rng);
    let xs: Vec<Vec<i64>> = (0..sl).map(|i| q512(&emb[i])).collect();
    let qv = q512(&emb[77]); let mut qn = qv.clone(); qn.resize(p.n, 0);
    let px = pack(&p, &xs, d);
    let mut tb = enc_template_b(&p, &s, &px, &mut rng);
    let mut ta = enc_template_a(&p, &s, &px, &mut rng);
    for t in 0..3 {
        let kp = key_polys(&p, &s);
        let qb = ctx.to_fourier(&enc_query_b(&p, &s, &qn, &mut rng));
        let mut o = tfhe::core_crypto::prelude::GlweCiphertext::new(0u64, p.glwe_size(), p.poly(), modulus());
        ctx.ext(&mut o, &qb, &tb);
        let qa = enc_query_a(&p, &s, &qn, &mut rng);
        let fa = ctx_to_f(&mut ctx, &ta); let oa = eval_a(&p, &mut ctx, &fa, &qa);
        for j in 0..sl {
            let ex = exact_ip(&xs[j], &qv) as f64;
            println!("t={t} slot {j}: exact {ex:9.0}  B {:11.2}  A {:11.2}", coef_score(&p, &kp, &o, j * d), coef_score(&p, &kp, &oa, j * d));
        }
        let s2 = keygen(&p, &mut rng);
        let tok = Token::from_std(&mut ctx, &token_gen(&p, &s, &s2, true, &mut rng));
        tb = rotate_glwe(&mut ctx, &tok, &tb); ta = rotate_ggsw(&p, &mut ctx, &tok, &ta); s = s2;
    }
}
fn ctx_to_f(c: &mut Ctx, g: &Ggsw) -> FGgsw { c.to_fourier(g) }
