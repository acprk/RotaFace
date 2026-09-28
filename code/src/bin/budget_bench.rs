//! Budget/cost trade-off of the compact pipeline (native seeded KS + re-randomisation + (36,32) rounding)
//! for token gadgets with bl*lv = 36. Measures single-core time per ciphertext (interleaved) and sigma_step
//! from 16 rotations (template noise growth), p128.
use rotaface::*;
use std::io::Write;
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let out = &a[1];
    let p = params_from_env();
    let gads = [(2usize, 18usize), (3, 12), (4, 9), (6, 6), (9, 4), (12, 3), (18, 2)];
    let emb: Vec<Vec<f32>> = load_f32(&data_path(), 512).into_iter().filter(|v| v.iter().map(|x| x * x).sum::<f32>() > 0.25).collect();
    let q512 = |v: &Vec<f32>| -> Vec<i64> { let mut w: Vec<i64> = v.iter().map(|&x| (x * p.scale).round() as i64).collect(); w.resize(p.n, 0); w };
    let xs: Vec<Vec<i64>> = (0..100).map(|i| q512(&emb[i])).collect();
    let mut rng = Rng::new(); let mut ctx = Ctx::new(&p);
    let s = keygen(&p, &mut rng); let s2 = keygen(&p, &mut rng);
    let pk = pk_gen(&p, &mut ctx, &s2, &mut rng);
    let cts: Vec<Glwe> = xs.iter().map(|x| { let mut c = enc_template_b(&p, &s, x, &mut rng); truncate_split(&p, &mut c, 36, 32); c }).collect();
    let nats: Vec<(usize, usize, NativeKsk, usize)> = gads.iter().map(|&(bl, lv)| { let mut pc = p; pc.r_base_log = bl; pc.r_level = lv; let sk = ksk_gen_seeded(&pc, &s, &s2, bl, lv); let b = sk.bytes(); (bl, lv, NativeKsk::expand(&pc, &mut ctx, &sk), b) }).collect();
    let mut times: Vec<Vec<f64>> = vec![vec![]; gads.len()];
    for _ in 0..7 { for (gi, (bl, lv, nat, _)) in nats.iter().enumerate() {
        let mut pc = p; pc.r_base_log = *bl; pc.r_level = *lv;
        let t0 = Instant::now();
        for c in &cts { let mut o = native_ks(&pc, &mut ctx, nat, c); rerandomise(&p, &mut ctx, &pk, &mut o); truncate_split(&p, &mut o, 36, 32); }
        times[gi].push(t0.elapsed().as_secs_f64() * 1e6 / cts.len() as f64);
    } }
    let mut f = std::fs::File::create(out).unwrap();
    writeln!(f, "bl,lv,us_per_ct_median,token_bytes,noise_log2_t16,sigma_step_log2").unwrap();
    for (gi, (bl, lv, _, tb)) in nats.iter().enumerate() {
        let mut v = times[gi].clone(); v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // noise growth over 16 rotations on 16 templates
        let mut pc = p; pc.r_base_log = *bl; pc.r_level = *lv;
        let mut sk_cur = keygen(&p, &mut rng);
        let mut tb16: Vec<Glwe> = xs[..16].iter().map(|x| { let mut c = enc_template_b(&p, &sk_cur, x, &mut rng); truncate_split(&p, &mut c, 36, 32); c }).collect();
        for _ in 0..16 {
            let sn = keygen(&p, &mut rng);
            let nat = NativeKsk::expand(&pc, &mut ctx, &ksk_gen_seeded(&pc, &sk_cur, &sn, *bl, *lv));
            let pkn = pk_gen(&p, &mut ctx, &sn, &mut rng);
            tb16 = tb16.iter().map(|c| { let mut o = native_ks(&pc, &mut ctx, &nat, c); rerandomise(&p, &mut ctx, &pkn, &mut o); truncate_split(&p, &mut o, 36, 32); o }).collect();
            sk_cur = sn;
        }
        let nz: Vec<f64> = tb16.iter().zip(&xs[..16]).flat_map(|(c, x)| glwe_noise(&p, &sk_cur, c, x)).collect();
        let s16 = std(&nz);
        let s0 = 3.2 * 2f64.powi(24);
        let step = ((s16 * s16 - s0 * s0) / 16.0).sqrt();
        writeln!(f, "{bl},{lv},{:.2},{tb},{:.3},{:.3}", v[3], s16.log2(), step.log2()).unwrap();
        println!("({bl},{lv}) {:.1} us/ct token {} B  noise@16 2^{:.2}  step 2^{:.2}", v[3], tb, s16.log2(), step.log2());
    }
}
