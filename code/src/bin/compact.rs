//! E11: compact rotation pipeline vs baseline (p128, layout B).
//! Timing: single core, interleaved rounds, median per ciphertext. Noise/score error along T rotations.
use rotaface::*;
use std::io::Write;
use std::time::Instant;

#[derive(Clone, Copy)]
struct Cfg { name: &'static str, native: bool, bl: usize, lv: usize, mask_bits: u32, body_bits: u32, rr: bool }

fn rotate(p: &Params, ctx: &mut Ctx, c: &Cfg, ext: &Token, nat: &NativeKsk, pk: &PubKey, ct: &Glwe) -> Glwe {
    let mut o = if c.native { native_ks(p, ctx, nat, ct) } else { rotate_glwe(ctx, ext, ct) };
    if c.rr { rerandomise(p, ctx, pk, &mut o); }
    truncate_split(p, &mut o, c.mask_bits, c.body_bits);
    o
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let t_max: usize = a[1].parse().unwrap(); let m: usize = a[2].parse().unwrap(); let nq: usize = a[3].parse().unwrap();
    let out = &a[4];
    let cfgs = [
        Cfg { name: "baseline_extprod_4x15_64b", native: false, bl: 4, lv: 15, mask_bits: 64, body_bits: 64, rr: false },
        Cfg { name: "native_4x15_64b", native: true, bl: 4, lv: 15, mask_bits: 64, body_bits: 64, rr: false },
        Cfg { name: "native_4x9_64b", native: true, bl: 4, lv: 9, mask_bits: 64, body_bits: 64, rr: false },
        Cfg { name: "compact_native_4x9_36_32", native: true, bl: 4, lv: 9, mask_bits: 36, body_bits: 32, rr: false },
        Cfg { name: "compact_native_4x9_36_32_rr", native: true, bl: 4, lv: 9, mask_bits: 36, body_bits: 32, rr: true },
    ];
    let base = params_from_env();
    let emb: Vec<Vec<f32>> = load_f32(&data_path(), 512).into_iter().filter(|v| v.iter().map(|x| x * x).sum::<f32>() > 0.25).collect();
    let q512 = |v: &Vec<f32>| -> Vec<i64> { let mut w: Vec<i64> = v.iter().map(|&x| (x * base.scale).round() as i64).collect(); w.resize(base.n, 0); w };
    let xs: Vec<Vec<i64>> = (0..m).map(|i| q512(&emb[i * 7])).collect();
    let qs: Vec<Vec<i64>> = (0..nq).map(|i| q512(&emb[5000 + i * 11])).collect();
    let mut f = std::fs::File::create(out).unwrap();
    writeln!(f, "config,kind,t,value1,value2,value3").unwrap();
    // ---- timing (interleaved), 100 ciphertexts, 7 rounds
    let mut rng = Rng::new();
    let tx: Vec<Vec<i64>> = (0..100).map(|i| q512(&emb[i])).collect();
    let mut setups = Vec::new();
    for c in &cfgs {
        let mut p = base; p.r_base_log = c.bl; p.r_level = c.lv;
        let mut ctx = Ctx::new(&p);
        let s = keygen(&p, &mut rng); let s2 = keygen(&p, &mut rng);
        let t0 = Instant::now(); let sk = ksk_gen_seeded(&p, &s, &s2, c.bl, c.lv); let tgen = t0.elapsed().as_secs_f64();
        let nat = NativeKsk::expand(&p, &mut ctx, &sk);
        let ext = Token::from_std(&mut ctx, &token_gen(&p, &s, &s2, false, &mut rng));
        let pk = pk_gen(&p, &mut ctx, &s2, &mut rng);
        let tok_bytes = if c.native { sk.bytes() + if c.rr { p.glwe_bytes() } else { 0 } } else { p.ggsw_bytes(c.lv) };
        let cts: Vec<Glwe> = tx.iter().map(|x| { let mut ct = enc_template_b(&p, &s, x, &mut rng); truncate_split(&p, &mut ct, c.mask_bits, c.body_bits); ct }).collect();
        writeln!(f, "{},sizes,0,{},{:.1},{:.3}", c.name, tok_bytes, stored_bytes_per_template(&p, c.mask_bits, c.body_bits), tgen * 1e3).unwrap();
        setups.push((p, ctx, ext, nat, pk, cts));
    }
    let mut times: Vec<Vec<f64>> = vec![vec![]; cfgs.len()];
    for _r in 0..7 {
        for (ci, c) in cfgs.iter().enumerate() {
            let (p, ctx, ext, nat, pk, cts) = &mut setups[ci];
            let t0 = Instant::now();
            for ct in cts.iter() { let _ = rotate(p, ctx, c, ext, nat, pk, ct); }
            times[ci].push(t0.elapsed().as_secs_f64() / cts.len() as f64 * 1e6);
        }
    }
    for (ci, c) in cfgs.iter().enumerate() {
        let mut v = times[ci].clone(); v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        writeln!(f, "{},time_us_per_ct,0,{:.2},{:.2},{:.2}", c.name, v[3], v[0], v[6]).unwrap();
        println!("{:32} rotate {:7.1} us/ct  ({:.1} us/template)", c.name, v[3], v[3] / base.slots(512) as f64);
    }
    // ---- noise along the chain
    for c in &cfgs {
        let mut p = base; p.r_base_log = c.bl; p.r_level = c.lv;
        let mut ctx = Ctx::new(&p);
        let mut s = keygen(&p, &mut rng);
        let mut tb: Vec<Glwe> = xs.iter().map(|x| { let mut ct = enc_template_b(&p, &s, x, &mut rng); truncate_split(&p, &mut ct, c.mask_bits, c.body_bits); ct }).collect();
        for t in 0..=t_max {
            if [0usize, 1, 16, 256, 1024].contains(&t) {
                let mut e = Vec::new(); let mut nz = Vec::new();
                for (i, x) in xs.iter().enumerate() {
                    nz.extend(glwe_noise(&p, &s, &tb[i], x));
                    for q in &qs { let qb = ctx.to_fourier(&enc_query_b(&p, &s, q, &mut rng)); e.push(score_b(&p, &mut ctx, &s, &qb, &tb[i]) - exact_ip(x, q) as f64); }
                }
                writeln!(f, "{},noise,{t},{:.3},{:.4},{:.2}", c.name, std(&nz).log2(), std(&e), e.iter().fold(0f64, |a, b| a.max(b.abs()))).unwrap();
                println!("{:32} t={t:5} noise 2^{:.2} err_std {:.2}", c.name, std(&nz).log2(), std(&e));
            }
            if t == t_max { break; }
            let s2 = keygen(&p, &mut rng);
            let sk = ksk_gen_seeded(&p, &s, &s2, c.bl, c.lv);
            let nat = NativeKsk::expand(&p, &mut ctx, &sk);
            let ext = if c.native { None } else { Some(Token::from_std(&mut ctx, &token_gen(&p, &s, &s2, false, &mut rng))) };
            let pk = pk_gen(&p, &mut ctx, &s2, &mut rng);
            tb = tb.iter().map(|ct| {
                let mut o = match &ext { Some(e) => rotate_glwe(&mut ctx, e, ct), None => native_ks(&p, &mut ctx, &nat, ct) };
                if c.rr { rerandomise(&p, &mut ctx, &pk, &mut o); }
                truncate_split(&p, &mut o, c.mask_bits, c.body_bits); o }).collect();
            s = s2;
        }
    }
}
