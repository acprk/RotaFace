//! E3: single-thread per-template costs (interleaved rounds, median), sizes.
//! usage: cost <rounds> <batch> <out.csv>
use rotaface::*;
use tfhe::core_crypto::prelude::GlweCiphertext;
use std::io::Write;
use std::time::Instant;

fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let rounds: usize = a[1].parse().unwrap();
    let batch: usize = a[2].parse().unwrap();
    let out = &a[3];
    let emb: Vec<Vec<f32>> = load_f32(&data_path(), 512).into_iter().filter(|v| v.iter().map(|x| x * x).sum::<f32>() > 0.25).collect();
    let base = params_from_env();
    let mut rng = Rng::new();
    let xs: Vec<Vec<i64>> = (0..batch).map(|i| quantise(&base, &emb[i])).collect();
    let q = quantise(&base, &emb[9000]);

    // configurations: (name, layout, q_base_log, q_level, r_base_log, r_level)
    let cfgs: Vec<(&str, char, usize, usize, usize, usize)> = vec![
        ("B_r4x15", 'B', 16, 2, 4, 15), ("B_r6x10", 'B', 16, 2, 6, 10), ("B_r8x7", 'B', 16, 2, 8, 7),
        ("B_r10x6", 'B', 16, 2, 10, 6), ("B_r12x5", 'B', 16, 2, 12, 5), ("B_r15x4", 'B', 16, 2, 15, 4),
        ("A_q16x2_r4x15", 'A', 16, 2, 4, 15), ("A_q8x4_r4x15", 'A', 8, 4, 4, 15), ("A_q8x4_r8x7", 'A', 8, 4, 8, 7),
        ("A_q4x8_r8x7", 'A', 4, 8, 8, 7), ("A_q4x8_r4x15", 'A', 4, 8, 4, 15),
    ];
    let cfgs: Vec<_> = match std::env::var("RF_CFG") { Ok(f) => cfgs.into_iter().filter(|c| c.0 == f).collect(), Err(_) => cfgs };
    let mut f = std::fs::File::create(out).unwrap();
    writeln!(f, "config,op,us_per_ct_median,us_min,us_max,rounds,batch,bytes_per_ct,slots").unwrap();
    struct St { p: Params, s: Sk, tok: Token, tok_bytes: usize, ta: Vec<Ggsw>, fa: Vec<FGgsw>, tb: Vec<Glwe>, qa: Glwe, qb: FGgsw, s2: Sk }
    let mut ctx0 = Ctx::new(&base);
    let mut st: Vec<(String, char, St)> = Vec::new();
    for (name, lay, qb_, ql, rb, rl) in &cfgs {
        let mut p = base; p.q_base_log = *qb_; p.q_level = *ql; p.r_base_log = *rb; p.r_level = *rl;
        let s = keygen(&p, &mut rng); let s2 = keygen(&p, &mut rng);
        let ts = token_gen(&p, &s, &s2, *lay == 'A', &mut rng);
        let tok_bytes = ts.bytes(&p);
        let tok = Token::from_std(&mut ctx0, &ts);
        let (ta, fa, tb) = if *lay == 'A' {
            let ta: Vec<Ggsw> = xs.iter().map(|x| enc_template_a(&p, &s, x, &mut rng)).collect();
            let fa = ta.iter().map(|g| ctx0.to_fourier(g)).collect();
            (ta, fa, vec![])
        } else { (vec![], vec![], xs.iter().map(|x| enc_template_b(&p, &s, x, &mut rng)).collect()) };
        let qa = enc_query_a(&p, &s, &q, &mut rng);
        let qb = ctx0.to_fourier(&enc_query_b(&p, &s, &q, &mut rng));
        st.push((name.to_string(), *lay, St { p, s, tok, tok_bytes, ta, fa, tb, qa, qb, s2 }));
    }
    // op name -> config -> samples
    let mut res: std::collections::BTreeMap<(String, String), (Vec<f64>, usize)> = Default::default();
    let mut push = |c: &str, op: &str, us: f64, bytes: usize| { res.entry((c.to_string(), op.to_string())).or_insert((vec![], bytes)).0.push(us); };
    for _r in 0..rounds {
        for (name, lay, x) in st.iter_mut() {
            let p = x.p; let mut ctx = Ctx::new(&p);
            if *lay == 'A' {
                let t0 = Instant::now(); let rot: Vec<Ggsw> = x.ta.iter().map(|g| rotate_ggsw(&p, &mut ctx, &x.tok, g)).collect();
                push(name, "server_rotate", t0.elapsed().as_secs_f64() * 1e6 / batch as f64, p.ggsw_bytes(p.q_level));
                let t0 = Instant::now(); let _f: Vec<FGgsw> = rot.iter().map(|g| ctx.to_fourier(g)).collect();
                push(name, "server_to_fourier", t0.elapsed().as_secs_f64() * 1e6 / batch as f64, p.fggsw_bytes(p.q_level));
                let t0 = Instant::now(); for g in &x.fa { let _ = eval_a(&p, &mut ctx, g, &x.qa); }
                push(name, "server_score", t0.elapsed().as_secs_f64() * 1e6 / batch as f64, p.glwe_bytes());
                let t0 = Instant::now();
                for g in &x.ta { let m = dec_template_a(&p, &x.s, g); let _ = enc_template_a(&p, &x.s2, &m, &mut rng); }
                push(name, "owner_reencrypt", t0.elapsed().as_secs_f64() * 1e6 / batch as f64, p.ggsw_bytes(p.q_level));
            } else {
                let t0 = Instant::now(); let _rot: Vec<Glwe> = x.tb.iter().map(|c| rotate_glwe(&mut ctx, &x.tok, c)).collect();
                push(name, "server_rotate", t0.elapsed().as_secs_f64() * 1e6 / batch as f64, p.glwe_bytes());
                let t0 = Instant::now(); for c in &x.tb { let mut o = GlweCiphertext::new(0u64, p.glwe_size(), p.poly(), modulus()); ctx.ext(&mut o, &x.qb, c); }
                push(name, "server_score", t0.elapsed().as_secs_f64() * 1e6 / batch as f64, p.glwe_bytes());
                let t0 = Instant::now();
                for c in &x.tb { let m = dec_template_b(&p, &x.s, c); let _ = enc_template_b(&p, &x.s2, &m, &mut rng); }
                push(name, "owner_reencrypt", t0.elapsed().as_secs_f64() * 1e6 / batch as f64, p.glwe_bytes());
            }
            let t0 = Instant::now(); let _ = token_gen(&p, &x.s, &x.s2, *lay == 'A', &mut rng);
            push(name, "owner_token_gen", t0.elapsed().as_secs_f64() * 1e6, x.tok_bytes);
            // client query encryption
            let t0 = Instant::now();
            if *lay == 'A' { let _ = enc_query_a(&p, &x.s, &q, &mut rng); push(name, "client_query_enc", t0.elapsed().as_secs_f64() * 1e6, p.glwe_bytes()); }
            else { let g = enc_query_b(&p, &x.s, &q, &mut rng); let _ = ctx.to_fourier(&g); push(name, "client_query_enc", t0.elapsed().as_secs_f64() * 1e6, p.ggsw_bytes(p.q_level)); }
        }
    }
    for ((c, op), (v, bytes)) in res {
        let mn = v.iter().cloned().fold(f64::MAX, f64::min); let mx = v.iter().cloned().fold(0.0, f64::max);
        writeln!(f, "{c},{op},{:.2},{:.2},{:.2},{},{},{},{}", median(v.clone()), mn, mx, rounds, batch, bytes, base.n / 512).unwrap();
        println!("{c:16} {op:18} {:10.2} us  bytes={bytes}", median(v));
    }
}
