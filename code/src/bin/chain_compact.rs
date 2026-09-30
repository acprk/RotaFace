//! Noise-model validation on the full rotation (native seeded KS + re-randomisation + (36,32) rounding).
//! usage: RF_PARAMS=p128 chain_compact <bl> <lv> <T> <M> <Q> <out.csv>
use rotaface::*;
use std::io::Write;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (bl, lv): (usize, usize) = (a[1].parse().unwrap(), a[2].parse().unwrap());
    let t_max: usize = a[3].parse().unwrap(); let m: usize = a[4].parse().unwrap(); let nq: usize = a[5].parse().unwrap();
    let out = &a[6];
    let mut p = params_from_env(); p.r_base_log = bl; p.r_level = lv;
    let emb: Vec<Vec<f32>> = load_f32(&data_path(), 512).into_iter().filter(|v| v.iter().map(|x| x * x).sum::<f32>() > 0.25).collect();
    let q512 = |v: &Vec<f32>| -> Vec<i64> { let mut w: Vec<i64> = v.iter().map(|&x| (x * p.scale).round() as i64).collect(); w.resize(p.n, 0); w };
    let xs: Vec<Vec<i64>> = (0..m).map(|i| q512(&emb[i * 7])).collect();
    let qs: Vec<Vec<i64>> = (0..nq).map(|i| q512(&emb[5000 + i * 11])).collect();
    let mut rng = Rng::new(); let mut ctx = Ctx::new(&p);
    let mut s = keygen(&p, &mut rng);
    let mut tb: Vec<Glwe> = xs.iter().map(|x| { let mut c = enc_template_b(&p, &s, x, &mut rng); truncate_split(&p, &mut c, 36, 32); c }).collect();
    let new_file = !std::path::Path::new(out).exists();
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(out).unwrap();
    if new_file { writeln!(f, "bl,lv,t,noise_log2,score_err_std,score_err_maxabs").unwrap(); }
    for t in 0..=t_max {
        if t == 0 || t.is_power_of_two() {
            let mut e = Vec::new(); let mut nz = Vec::new();
            for (i, x) in xs.iter().enumerate() {
                nz.extend(glwe_noise(&p, &s, &tb[i], x));
                for q in &qs { let qb = ctx.to_fourier(&enc_query_b(&p, &s, q, &mut rng)); e.push(score_b(&p, &mut ctx, &s, &qb, &tb[i]) - exact_ip(x, q) as f64); }
            }
            writeln!(f, "{bl},{lv},{t},{:.3},{:.4},{:.2}", std(&nz).log2(), std(&e), e.iter().fold(0f64, |a, b| a.max(b.abs()))).unwrap();
            println!("({bl},{lv}) t={t} noise 2^{:.2} err {:.2}", std(&nz).log2(), std(&e));
        }
        if t == t_max { break; }
        let s2 = keygen(&p, &mut rng);
        let nat = NativeKsk::expand(&p, &mut ctx, &ksk_gen_seeded(&p, &s, &s2, bl, lv));
        let pk = pk_gen(&p, &mut ctx, &s2, &mut rng);
        tb = tb.iter().map(|c| { let mut o = native_ks(&p, &mut ctx, &nat, c); rerandomise(&p, &mut ctx, &pk, &mut o); truncate_split(&p, &mut o, 36, 32); o }).collect();
        s = s2;
    }
}
