//! E6: in-place hardness upgrade of a legacy BioVite-shape DB (k=3,N=512, ~2^96) to k=4,N=512 (~2^140),
//! then continued rotation in the new parameters. Reports score error + per-template cost.
//! usage: upgrade <A|B> <qB> <qL> <rB> <rL> <t_after> <M> <Q> <out.csv>
use rotaface::*;
use std::io::Write;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let lay = a[1].chars().next().unwrap();
    let mut p = Params::biovite(); let mut p2 = Params::p128k4();
    for q in [&mut p, &mut p2] { q.q_base_log = a[2].parse().unwrap(); q.q_level = a[3].parse().unwrap(); q.r_base_log = a[4].parse().unwrap(); q.r_level = a[5].parse().unwrap(); }
    let t_after: usize = a[6].parse().unwrap(); let m: usize = a[7].parse().unwrap(); let nq: usize = a[8].parse().unwrap();
    let out = &a[9];
    let emb: Vec<Vec<f32>> = load_f32(&data_path(), 512).into_iter().filter(|v| v.iter().map(|x| x * x).sum::<f32>() > 0.25).collect();
    let xs: Vec<Vec<i64>> = (0..m).map(|i| quantise(&p, &emb[i * 13])).collect();
    let qs: Vec<Vec<i64>> = (0..nq).map(|i| quantise(&p, &emb[6000 + i * 17])).collect();
    let mut rng = Rng::new();
    let s = keygen(&p, &mut rng);
    let mut ctx = Ctx::new(&p); let mut ctx2 = Ctx::new(&p2);
    let new_file = !std::path::Path::new(out).exists();
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(out).unwrap();
    if new_file { writeln!(f, "layout,q_base_log,q_level,r_base_log,r_level,stage,k,t,score_err_std,score_err_maxabs,us_per_template").unwrap(); }
    let eval = |pp: &Params, c: &mut Ctx, sk: &Sk, ta: &[Ggsw], tb: &[Glwe], rng: &mut Rng| -> (f64, f64) {
        let mut e = Vec::new();
        for (i, x) in xs.iter().enumerate() { for q in &qs {
            let ex = exact_ip(x, q) as f64;
            if lay == 'A' { let fa = c.to_fourier(&ta[i]); let qa = enc_query_a(pp, sk, q, rng); e.push(score_a(pp, c, sk, &fa, &qa) - ex); }
            else { let qb = c.to_fourier(&enc_query_b(pp, sk, q, rng)); e.push(score_b(pp, c, sk, &qb, &tb[i]) - ex); }
        }}
        (std(&e), e.iter().fold(0f64, |acc, v| acc.max(v.abs())))
    };
    let (mut ta, mut tb): (Vec<Ggsw>, Vec<Glwe>) = if lay == 'A' { (xs.iter().map(|x| enc_template_a(&p, &s, x, &mut rng)).collect(), vec![]) } else { (vec![], xs.iter().map(|x| enc_template_b(&p, &s, x, &mut rng)).collect()) };
    let (es, em) = eval(&p, &mut ctx, &s, &ta, &tb, &mut rng);
    writeln!(f, "{lay},{},{},{},{},legacy,{},0,{es:.3},{em:.2},", p.q_base_log, p.q_level, p.r_base_log, p.r_level, p.k).unwrap();
    println!("legacy k=3: err std {es:.3} max {em:.1}");
    // upgrade
    let mut s2 = keygen(&p2, &mut rng);
    let tok = Token::from_std(&mut ctx2, &token_gen(&p2, &pad_key(&p, p2.k, &s), &s2, lay == 'A', &mut rng));
    let t0 = Instant::now();
    if lay == 'A' { ta = ta.iter().map(|g| upgrade_ggsw(&p, &p2, &mut ctx2, &tok, g)).collect(); } else { tb = tb.iter().map(|c| upgrade_glwe(&p, &p2, &mut ctx2, &tok, c)).collect(); }
    let us = t0.elapsed().as_secs_f64() * 1e6 / m as f64;
    let (es, em) = eval(&p2, &mut ctx2, &s2, &ta, &tb, &mut rng);
    writeln!(f, "{lay},{},{},{},{},upgraded,{},1,{es:.3},{em:.2},{us:.1}", p.q_base_log, p.q_level, p.r_base_log, p.r_level, p2.k).unwrap();
    println!("upgraded k=4: err std {es:.3} max {em:.1}  ({us:.0} us/template)");
    for t in 1..=t_after {
        let s3 = keygen(&p2, &mut rng);
        let tok = Token::from_std(&mut ctx2, &token_gen(&p2, &s2, &s3, lay == 'A', &mut rng));
        if lay == 'A' { ta = ta.iter().map(|g| rotate_ggsw(&p2, &mut ctx2, &tok, g)).collect(); } else { tb = tb.iter().map(|c| rotate_glwe(&mut ctx2, &tok, c)).collect(); }
        s2 = s3;
        if t.is_power_of_two() {
            let (es, em) = eval(&p2, &mut ctx2, &s2, &ta, &tb, &mut rng);
            writeln!(f, "{lay},{},{},{},{},rotated,{},{},{es:.3},{em:.2},", p.q_base_log, p.q_level, p.r_base_log, p.r_level, p2.k, t + 1).unwrap();
            println!("after upgrade + {t} rotations: err std {es:.3} max {em:.1}");
        }
    }
}
