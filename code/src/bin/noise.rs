//! E1: noise / score-error growth along the rotation chain.
//! usage: noise <layout A|B> <q_base_log> <q_level> <r_base_log> <r_level> <T> <M> <Q> <out.csv> [raw_prefix]
use rotaface::*;
use std::io::Write;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let layout = a[1].clone();
    let mut p = params_from_env();
    p.q_base_log = a[2].parse().unwrap(); p.q_level = a[3].parse().unwrap();
    p.r_base_log = a[4].parse().unwrap(); p.r_level = a[5].parse().unwrap();
    let t_max: usize = a[6].parse().unwrap();
    let m: usize = a[7].parse().unwrap();
    let nq: usize = a[8].parse().unwrap();
    let out = &a[9];
    let raw = a.get(10).cloned();
    let emb: Vec<Vec<f32>> = load_f32(&rotaface::data_path(), 512).into_iter()
        .filter(|v| v.iter().map(|x| x * x).sum::<f32>() > 0.25).collect();
    let mut rng = Rng::new();
    let mut ctx = Ctx::new(&p);
    let mut s = keygen(&p, &mut rng);
    let xs: Vec<Vec<i64>> = (0..m).map(|i| quantise(&p, &emb[i * 7])).collect();
    let qs: Vec<Vec<i64>> = (0..nq).map(|i| quantise(&p, &emb[5000 + i * 11])).collect();
    let is_a = layout == "A";
    let mut ta: Vec<Ggsw> = if is_a { xs.iter().map(|x| enc_template_a(&p, &s, x, &mut rng)).collect() } else { vec![] };
    let mut tb: Vec<Glwe> = if !is_a { xs.iter().map(|x| enc_template_b(&p, &s, x, &mut rng)).collect() } else { vec![] };
    let new_file = !std::path::Path::new(out).exists();
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(out).unwrap();
    if new_file { writeln!(f, "layout,q_base_log,q_level,r_base_log,r_level,t,body_log2,mask_log2,score_err_std,score_err_maxabs,n_scores").unwrap(); }
    let mut checkpoints: Vec<usize> = vec![0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
    checkpoints.retain(|&c| c <= t_max);
    for t in 0..=t_max {
        if checkpoints.contains(&t) {
            let mut errs = Vec::new();
            let (mut body, mut mask) = (Vec::new(), Vec::new());
            for (i, x) in xs.iter().enumerate() {
                if is_a {
                    let fa = ctx.to_fourier(&ta[i]);
                    for q in &qs { let qa = enc_query_a(&p, &s, q, &mut rng); errs.push(score_a(&p, &mut ctx, &s, &fa, &qa) - exact_ip(x, q) as f64); }
                    let (b, mk) = ggsw_row_noise(&p, &s, &ta[i], x); body.extend(b); mask.extend(mk);
                } else {
                    for q in &qs { let qb = ctx.to_fourier(&enc_query_b(&p, &s, q, &mut rng)); errs.push(score_b(&p, &mut ctx, &s, &qb, &tb[i]) - exact_ip(x, q) as f64); }
                    body.extend(glwe_noise(&p, &s, &tb[i], x));
                }
            }
            let mx = errs.iter().fold(0f64, |acc, e| acc.max(e.abs()));
            let ml = if mask.is_empty() { f64::NAN } else { std(&mask).log2() };
            writeln!(f, "{},{},{},{},{},{},{:.3},{:.3},{:.4},{:.3},{}", layout, p.q_base_log, p.q_level, p.r_base_log, p.r_level, t, std(&body).log2(), ml, std(&errs), mx, errs.len()).unwrap();
            println!("{layout} t={t} body={:.2} mask={:.2} err_std={:.3} max={:.1}", std(&body).log2(), ml, std(&errs), mx);
            if let Some(r) = &raw {
                let mut g = std::fs::File::create(format!("{r}_{layout}_t{t}.txt")).unwrap();
                for e in &errs { writeln!(g, "{e}").unwrap(); }
            }
        }
        if t == t_max { break; }
        let s2 = keygen(&p, &mut rng);
        let tok = Token::from_std(&mut ctx, &token_gen(&p, &s, &s2, is_a, &mut rng));
        if is_a { ta = ta.iter().map(|g| rotate_ggsw(&p, &mut ctx, &tok, g)).collect(); }
        else { tb = tb.iter().map(|c| rotate_glwe(&mut ctx, &tok, c)).collect(); }
        s = s2;
    }
}
