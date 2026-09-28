//! E10: noise-adaptive template compression along the rotation chain (layout B).
//! usage: RF_PARAMS=p128 compress <rB> <rL> <T> <M> <Q> <policy: none|fixed:<drop>|adaptive:<frac>> <out.csv>
use rotaface::*;
use std::io::Write;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut p = params_from_env();
    p.r_base_log = a[1].parse().unwrap(); p.r_level = a[2].parse().unwrap();
    let t_max: usize = a[3].parse().unwrap(); let m: usize = a[4].parse().unwrap(); let nq: usize = a[5].parse().unwrap();
    let policy = a[6].clone(); let out = &a[7];
    let emb: Vec<Vec<f32>> = load_f32(&data_path(), 512).into_iter().filter(|v| v.iter().map(|x| x * x).sum::<f32>() > 0.25).collect();
    let q512 = |v: &Vec<f32>| -> Vec<i64> { let mut w: Vec<i64> = v.iter().map(|&x| (x * p.scale).round() as i64).collect(); w.resize(p.n, 0); w };
    let xs: Vec<Vec<i64>> = (0..m).map(|i| q512(&emb[i * 7])).collect();
    let qs: Vec<Vec<i64>> = (0..nq).map(|i| q512(&emb[5000 + i * 11])).collect();
    let mut rng = Rng::new(); let mut ctx = Ctx::new(&p);
    let mut s = keygen(&p, &mut rng);
    let mut tb: Vec<Glwe> = xs.iter().map(|x| enc_template_b(&p, &s, x, &mut rng)).collect();
    let drop_for = |t: usize| -> u32 {
        if policy == "none" { 0 } else if let Some(d) = policy.strip_prefix("fixed:") { d.parse().unwrap() }
        else { let f: f64 = policy.strip_prefix("adaptive:").unwrap().parse().unwrap(); adaptive_drop(&p, model_sigma_t(&p, t), f) }
    };
    let new_file = !std::path::Path::new(out).exists();
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(out).unwrap();
    if new_file { writeln!(f, "policy,r_base_log,r_level,t,drop_bits,kept_bits,bytes_per_template,noise_log2,score_err_std,score_err_maxabs").unwrap(); }
    let d0 = drop_for(0); for c in tb.iter_mut() { truncate_glwe(c, d0); }
    let cps = [0usize, 1, 4, 16, 64, 256, 1024];
    for t in 0..=t_max {
        let d = drop_for(t);
        if cps.contains(&t) {
            let mut e = Vec::new(); let mut nz = Vec::new();
            for (i, x) in xs.iter().enumerate() {
                nz.extend(glwe_noise(&p, &s, &tb[i], x));
                for q in &qs { let qb = ctx.to_fourier(&enc_query_b(&p, &s, q, &mut rng)); e.push(score_b(&p, &mut ctx, &s, &qb, &tb[i]) - exact_ip(x, q) as f64); }
            }
            let kept = 64 - d; let bytes = (p.k + 1) * p.n * kept as usize / 8 / p.slots(512);
            let mx = e.iter().fold(0f64, |a, b| a.max(b.abs()));
            writeln!(f, "{policy},{},{},{t},{d},{kept},{bytes},{:.3},{:.4},{:.2}", p.r_base_log, p.r_level, std(&nz).log2(), std(&e), mx).unwrap();
            println!("{policy} t={t} drop={d} kept={kept} bytes/tmpl={bytes} noise=2^{:.2} err_std={:.2}", std(&nz).log2(), std(&e));
        }
        if t == t_max { break; }
        let s2 = keygen(&p, &mut rng);
        let tok = Token::from_std(&mut ctx, &token_gen(&p, &s, &s2, false, &mut rng));
        let dn = drop_for(t + 1);
        tb = tb.iter().map(|c| { let mut o = rotate_glwe(&mut ctx, &tok, c); truncate_glwe(&mut o, dn); o }).collect();
        s = s2;
    }
}
