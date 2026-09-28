//! Sanity check: encrypt, score, rotate a few times, score again (layouts A and B).
use rotaface::*;

fn main() {
    let mut p = Params::biovite();
    let a: Vec<usize> = std::env::args().skip(1).map(|x| x.parse().unwrap()).collect();
    if a.len() == 2 { p.r_base_log = a[0]; p.r_level = a[1]; }
    println!("{:?}", p);
    let emb: Vec<Vec<f32>> = load_f32(&rotaface::data_path(), 512).into_iter()
        .filter(|v| v.iter().map(|x| x * x).sum::<f32>() > 0.25).collect();
    let mut rng = Rng::new();
    let mut ctx = Ctx::new(&p);
    let mut s = keygen(&p, &mut rng);
    let m = 8;
    let xs: Vec<Vec<i64>> = (0..m).map(|i| quantise(&p, &emb[i])).collect();
    let qs: Vec<Vec<i64>> = (0..m).map(|i| quantise(&p, &emb[100 + i])).collect();

    let mut ta: Vec<Ggsw> = xs.iter().map(|x| enc_template_a(&p, &s, x, &mut rng)).collect();
    let mut tb: Vec<Glwe> = xs.iter().map(|x| enc_template_b(&p, &s, x, &mut rng)).collect();

    for t in 0..=4 {
        let (mut ea, mut eb) = (Vec::new(), Vec::new());
        for (i, x) in xs.iter().enumerate() {
            let fa = ctx.to_fourier(&ta[i]);
            for q in &qs {
                let exact = exact_ip(x, q) as f64;
                let qa = enc_query_a(&p, &s, q, &mut rng);
                ea.push(score_a(&p, &mut ctx, &s, &fa, &qa) - exact);
                let qb = ctx.to_fourier(&enc_query_b(&p, &s, q, &mut rng));
                eb.push(score_b(&p, &mut ctx, &s, &qb, &tb[i]) - exact);
            }
        }
        let (bn, mn) = ggsw_row_noise(&p, &s, &ta[0], &xs[0]);
        let gn = glwe_noise(&p, &s, &tb[0], &xs[0]);
        let mx = |v: &Vec<f64>| v.iter().fold(0f64, |a, b| a.max(b.abs()));
        println!("t={t}  A: score err std {:.3} max {:.3} | rows log2 std body {:.2} mask {:.2} || B: score err std {:.3} max {:.3} | glwe log2 std {:.2}",
                 std(&ea), mx(&ea), std(&bn).log2(), std(&mn).log2(), std(&eb), mx(&eb), std(&gn).log2());
        // rotate
        let s2 = keygen(&p, &mut rng);
        let tok_std = token_gen(&p, &s, &s2, true, &mut rng);
        let tok = Token::from_std(&mut ctx, &tok_std);
        ta = ta.iter().map(|g| rotate_ggsw(&p, &mut ctx, &tok, g)).collect();
        tb = tb.iter().map(|c| rotate_glwe(&mut ctx, &tok, c)).collect();
        s = s2;
    }
    // decryption baseline check
    let xa = dec_template_a(&p, &s, &ta[0]);
    let xb = dec_template_b(&p, &s, &tb[0]);
    println!("template recovery after 5 rotations: A ok={} B ok={}", xa == xs[0], xb == xs[0]);
}
