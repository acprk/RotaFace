//! E2: full encrypted 1:N identification on LFW after t rotations (no noise simulation).
//! usage: RF_PARAMS=<p128|biovite> ident <A|B> <qB> <qL> <rB> <rL> <t_list comma> <threads> <out_prefix>
//! Templates are slot-packed (n/512 per ciphertext). Writes <out_prefix>_t<t>.f32 :
//! probes x gallery decrypted scores / scale^2, and <out_prefix>_quant.f32 (exact quantised).
use rayon::prelude::*;
use rotaface::*;
use std::io::Write;
use tfhe::core_crypto::prelude::GlweCiphertext;

const D: usize = 512;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let lay = a[1].chars().next().unwrap();
    let mut p = params_from_env();
    p.q_base_log = a[2].parse().unwrap(); p.q_level = a[3].parse().unwrap();
    p.r_base_log = a[4].parse().unwrap(); p.r_level = a[5].parse().unwrap();
    let ts: Vec<usize> = a[6].split(',').map(|x| x.parse().unwrap()).collect();
    let th: usize = a[7].parse().unwrap();
    let pre = &a[8];
    rayon::ThreadPoolBuilder::new().num_threads(th).build_global().unwrap();
    let sl = p.slots(D);
    let emb = load_f32(&data_path(), D);
    let split = std::fs::read_to_string(data_path().replace("lfw_emb.f32", "split.txt")).unwrap();
    let lines: Vec<Vec<usize>> = split.lines().map(|l| l.split_whitespace().map(|x| x.parse().unwrap()).collect()).collect();
    let (gal, prb) = (&lines[0], &lines[1]);
    let q512 = |v: &Vec<f32>| -> Vec<i64> { v.iter().map(|&x| (x * p.scale).round() as i64).collect() };
    let xs: Vec<Vec<i64>> = gal.iter().map(|&i| q512(&emb[i])).collect();
    let qs: Vec<Vec<i64>> = prb.iter().map(|&i| q512(&emb[i])).collect();
    let ng = xs.len();
    let sc2 = (p.scale * p.scale) as f64;
    {
        let mut f = std::fs::File::create(format!("{pre}_quant.f32")).unwrap();
        let rows: Vec<Vec<f32>> = qs.par_iter().map(|q| xs.iter().map(|x| (exact_ip(x, q) as f64 / sc2) as f32).collect()).collect();
        for r in rows { for v in r { f.write_all(&v.to_le_bytes()).unwrap(); } }
    }
    // pack gallery: ciphertext c holds gallery items c*sl .. c*sl+sl-1
    let packed: Vec<Vec<i64>> = xs.chunks(sl).map(|ch| pack(&p, ch, D)).collect();
    let nct = packed.len();
    eprintln!("params k={} N={} slots={sl} gallery={ng} ciphertexts={nct} probes={}", p.k, p.n, qs.len());
    let mut rng = Rng::new();
    let mut s = keygen(&p, &mut rng);
    let mut ta: Vec<Ggsw> = vec![]; let mut tb: Vec<Glwe> = vec![];
    if lay == 'A' { ta = packed.par_iter().map(|x| { let mut r = Rng::new(); enc_template_a(&p, &s, x, &mut r) }).collect(); }
    else { let d0: u32 = std::env::var("RF_DROP").ok().map(|v| v.parse().unwrap()).unwrap_or(0);
           tb = packed.par_iter().map(|x| { let mut r = Rng::new(); let mut c = enc_template_b(&p, &s, x, &mut r); truncate_glwe(&mut c, d0);
               if std::env::var("RF_COMPACT").is_ok() { truncate_split(&p, &mut c, 36, 32); } c }).collect(); }
    let tmax = *ts.iter().max().unwrap();
    let mut ctx = Ctx::new(&p);
    let mut log = std::fs::OpenOptions::new().create(true).append(true).open(format!("{pre}_timing.csv")).unwrap();
    for t in 0..=tmax {
        if ts.contains(&t) {
            let t0 = std::time::Instant::now();
            let kp = key_polys(&p, &s);
            let decode = |o: &Glwe, c: usize, row: &mut Vec<f32>| {
                for j in 0..sl { if c * sl + j < ng { row[c * sl + j] = (coef_score(&p, &kp, o, j * D) / sc2) as f32; } }
            };
            let rows: Vec<Vec<f32>> = if lay == 'A' {
                let fa: Vec<FGgsw> = ta.par_iter().map_init(|| Ctx::new(&p), |c, g| c.to_fourier(g)).collect();
                qs.par_iter().map_init(|| (Ctx::new(&p), Rng::new()), |(c, r), q| {
                    let mut qn = q.clone(); qn.resize(p.n, 0);
                    let qa = enc_query_a(&p, &s, &qn, r);
                    let mut row = vec![0f32; ng];
                    for (ci, g) in fa.iter().enumerate() { let o = eval_a(&p, c, g, &qa); decode(&o, ci, &mut row); }
                    row
                }).collect()
            } else {
                qs.par_iter().map_init(|| (Ctx::new(&p), Rng::new()), |(c, r), q| {
                    let mut qn = q.clone(); qn.resize(p.n, 0);
                    let qb = c.to_fourier(&enc_query_b(&p, &s, &qn, r));
                    let mut row = vec![0f32; ng];
                    for (ci, x) in tb.iter().enumerate() {
                        let mut o = GlweCiphertext::new(0u64, p.glwe_size(), p.poly(), modulus());
                        c.ext(&mut o, &qb, x); decode(&o, ci, &mut row);
                    }
                    row
                }).collect()
            };
            let el = t0.elapsed().as_secs_f64();
            let mut f = std::fs::File::create(format!("{pre}_t{t}.f32")).unwrap();
            for r in rows { for v in r { f.write_all(&v.to_le_bytes()).unwrap(); } }
            writeln!(log, "{lay},{t},{},{},{el:.2}", qs.len(), ng).unwrap();
            eprintln!("{lay} t={t} scored {}x{} in {:.1}s", qs.len(), ng, el);
        }
        if t == tmax { break; }
        let s2 = keygen(&p, &mut rng);
        let tok = Token::from_std(&mut ctx, &token_gen(&p, &s, &s2, lay == 'A', &mut rng));
        let rr = std::env::var("RF_RERAND").is_ok();
        let drop: u32 = std::env::var("RF_DROP").ok().map(|v| v.parse().unwrap()).unwrap_or(0);
        let pk = if rr { Some(pk_gen(&p, &mut ctx, &s2, &mut rng)) } else { None };
        if lay == 'A' { ta = ta.par_iter().map_init(|| Ctx::new(&p), |c, g| rotate_ggsw(&p, c, &tok, g)).collect(); }
        else if std::env::var("RF_COMPACT").is_ok() {
            let (gb, gl): (usize, usize) = std::env::var("RF_GADGET").ok().map(|v| { let w: Vec<usize> = v.split(',').map(|x| x.parse().unwrap()).collect(); (w[0], w[1]) }).unwrap_or((4, 9));
            let mut pc = p; pc.r_base_log = gb; pc.r_level = gl;
            let sk = ksk_gen_seeded(&pc, &s, &s2, gb, gl);
            let nat = NativeKsk::expand(&pc, &mut ctx, &sk);
            tb = tb.par_iter().map_init(|| Ctx::new(&p), |c, g| {
                let mut o = native_ks(&pc, c, &nat, g);
                if let Some(k) = &pk { rerandomise(&p, c, k, &mut o); }
                truncate_split(&p, &mut o, 36, 32); o }).collect(); }
        else { tb = tb.par_iter().map_init(|| Ctx::new(&p), |c, g| {
            let mut o = rotate_glwe(c, &tok, g);
            if let Some(k) = &pk { rerandomise(&p, c, k, &mut o); }
            truncate_glwe(&mut o, drop); o }).collect(); }
        s = s2;
    }
}
