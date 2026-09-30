//! E4: parallel rotation throughput vs thread count (rayon, work-stealing, per-thread FFT ctx).
//! usage: scale <A|B> <q_base_log> <q_level> <r_base_log> <r_level> <n_templates> <threads,comma> <repeats> <out.csv>
use rayon::prelude::*;
use rotaface::*;
use std::io::Write;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let lay = a[1].chars().next().unwrap();
    let mut p = params_from_env();
    p.q_base_log = a[2].parse().unwrap(); p.q_level = a[3].parse().unwrap();
    p.r_base_log = a[4].parse().unwrap(); p.r_level = a[5].parse().unwrap();
    let n: usize = a[6].parse().unwrap();
    let threads: Vec<usize> = a[7].split(',').map(|x| x.parse().unwrap()).collect();
    let reps: usize = a[8].parse().unwrap();
    let out = &a[9];
    let emb: Vec<Vec<f32>> = load_f32(&data_path(), 512).into_iter().filter(|v| v.iter().map(|x| x * x).sum::<f32>() > 0.25).collect();
    let mut rng = Rng::new();
    let s = keygen(&p, &mut rng); let s2 = keygen(&p, &mut rng);
    let mut ctx = Ctx::new(&p);
    let tok = Token::from_std(&mut ctx, &token_gen(&p, &s, &s2, lay == 'A', &mut rng));
    let compact = if std::env::var("RF_COMPACT").is_ok() {
        let mut pc = p; pc.r_level = 9;
        let sk = ksk_gen_seeded(&pc, &s, &s2, 4, 9);
        Some((NativeKsk::expand(&pc, &mut ctx, &sk), pk_gen(&p, &mut ctx, &s2, &mut rng), pc))
    } else { None };
    // build a DB of n templates: encrypt |emb| distinct ones, replicate (ciphertext content does not affect cost)
    let uniq = emb.len().min(n).min(2000);
    let xs: Vec<Vec<i64>> = (0..uniq).map(|i| quantise(&p, &emb[i])).collect();
    eprintln!("building DB n={n} ...");
    let mut ta: Vec<Ggsw> = vec![]; let mut tb: Vec<Glwe> = vec![];
    if lay == 'A' { let base: Vec<Ggsw> = xs.iter().map(|x| enc_template_a(&p, &s, x, &mut rng)).collect(); ta = (0..n).map(|i| base[i % uniq].clone()).collect(); }
    else {
        let base: Vec<Glwe> = xs.iter().map(|x| enc_template_b(&p, &s, x, &mut rng)).collect();
        // RF_FIRST_TOUCH=1: allocate the DB from all threads so that pages spread over both NUMA nodes
        tb = if std::env::var("RF_FIRST_TOUCH").is_ok() { (0..n).into_par_iter().map(|i| base[i % uniq].clone()).collect() }
             else { (0..n).map(|i| base[i % uniq].clone()).collect() };
    }
    // RF_PERM=1: write rotated ciphertexts in a fresh random order (breaks linking by storage position)
    let perm: Option<Vec<usize>> = if std::env::var("RF_PERM").is_ok() {
        let mut v: Vec<usize> = (0..n).collect(); use rand::seq::SliceRandom; v.shuffle(&mut rand::thread_rng()); Some(v) } else { None };
    let new_file = !std::path::Path::new(out).exists();
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(out).unwrap();
    if new_file { writeln!(f, "layout,q_base_log,q_level,r_base_log,r_level,n,threads,rep,seconds,templates_per_s,tag").unwrap(); }
    let tag = std::env::var("RF_TAG").unwrap_or_default();
    for rep in 0..reps {
        for &th in &threads {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(th).build().unwrap();
            let t0 = Instant::now();
            pool.install(|| {
                if lay == 'A' {
                    let _r: Vec<Ggsw> = ta.par_iter().map_init(|| Ctx::new(&p), |c, g| rotate_ggsw(&p, c, &tok, g)).collect();
                } else if let (Some((nat, pk, pc)), Some(pm)) = (&compact, &perm) {
                    let _r: Vec<Glwe> = (0..n).into_par_iter().map_init(|| Ctx::new(&p), |c, i| {
                        let mut o = native_ks(pc, c, nat, &tb[pm[i]]); rerandomise(&p, c, pk, &mut o); truncate_split(&p, &mut o, 36, 32); o }).collect();
                } else if let Some((nat, pk, pc)) = &compact {
                    let _r: Vec<Glwe> = tb.par_iter().map_init(|| Ctx::new(&p), |c, g| {
                        let mut o = native_ks(pc, c, nat, g); rerandomise(&p, c, pk, &mut o); truncate_split(&p, &mut o, 36, 32); o }).collect();
                } else {
                    let _r: Vec<Glwe> = tb.par_iter().map_init(|| Ctx::new(&p), |c, g| rotate_glwe(c, &tok, g)).collect();
                }
            });
            let sec = t0.elapsed().as_secs_f64();
            writeln!(f, "{lay},{},{},{},{},{n},{th},{rep},{sec:.4},{:.1},{tag}", p.q_base_log, p.q_level, p.r_base_log, p.r_level, n as f64 / sec).unwrap();
            println!("{lay} threads={th:3} rep={rep} {sec:8.3}s  {:10.1} tmpl/s", n as f64 / sec);
        }
    }
}
