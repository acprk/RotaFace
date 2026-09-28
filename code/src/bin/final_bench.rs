//! Final single-source benchmark (p128, layout B): all per-ciphertext costs measured in ONE interleaved run.
//! usage: RF_PARAMS=p128 final_bench <rounds> <batch> <out.csv>
use rotaface::*;
use std::io::Write;
use std::time::Instant;
use tfhe::core_crypto::prelude::GlweCiphertext;
fn med(mut v: Vec<f64>) -> (f64, f64, f64) { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); (v[v.len() / 2], v[0], v[v.len() - 1]) }
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let rounds: usize = a[1].parse().unwrap(); let batch: usize = a[2].parse().unwrap();
    let mut p = params_from_env(); p.r_base_log = 4; p.r_level = 15;
    let mut pc = p; pc.r_level = 9;
    let emb: Vec<Vec<f32>> = load_f32(&data_path(), 512).into_iter().filter(|v| v.iter().map(|x| x * x).sum::<f32>() > 0.25).collect();
    let q512 = |v: &Vec<f32>| -> Vec<i64> { let mut w: Vec<i64> = v.iter().map(|&x| (x * p.scale).round() as i64).collect(); w.resize(p.n, 0); w };
    let xs: Vec<Vec<i64>> = (0..batch * 4).step_by(4).map(|i| pack(&p, &[q512(&emb[i]), q512(&emb[i+1]), q512(&emb[i+2]), q512(&emb[i+3])].iter().map(|v| v[..512].to_vec()).collect::<Vec<_>>(), 512)).collect();
    let q = q512(&emb[9000]);
    let mut rng = Rng::new(); let mut ctx = Ctx::new(&p);
    let s = keygen(&p, &mut rng); let s2 = keygen(&p, &mut rng);
    let ext = Token::from_std(&mut ctx, &token_gen(&p, &s, &s2, false, &mut rng));
    let sk = ksk_gen_seeded(&pc, &s, &s2, 4, 9); let nat = NativeKsk::expand(&pc, &mut ctx, &sk);
    let sk15 = ksk_gen_seeded(&p, &s, &s2, 4, 15); let nat15 = NativeKsk::expand(&p, &mut ctx, &sk15);
    let pk = pk_gen(&p, &mut ctx, &s2, &mut rng);
    let fk1 = fkey(&p, &mut ctx, &s); let fk2 = fkey(&p, &mut ctx, &s2);
    let full: Vec<Glwe> = xs.iter().map(|x| enc_template_b(&p, &s, x, &mut rng)).collect();
    let comp: Vec<Glwe> = full.iter().map(|c| { let mut c = c.clone(); truncate_split(&p, &mut c, 36, 32); c }).collect();
    let qb = ctx.to_fourier(&enc_query_b(&p, &s, &q, &mut rng));
    let ops = ["rotate_extprod_4x15_64b", "rotate_compact_4x9", "rotate_compact_4x9_rerand", "owner_decrypt_reencrypt", "server_match", "client_query_ggsw", "owner_token_gen_seeded_plus_pk", "rotate_native_4x15_64b", "owner_decrypt_reencrypt_fft", "rotate_native_4x9_64b"];
    let mut t: Vec<Vec<f64>> = vec![vec![]; ops.len()];
    for _ in 0..rounds {
        let t0 = Instant::now(); for c in &full { let _ = rotate_glwe(&mut ctx, &ext, c); } t[0].push(t0.elapsed().as_secs_f64() * 1e6 / batch as f64);
        let t0 = Instant::now(); for c in &comp { let mut o = native_ks(&pc, &mut ctx, &nat, c); truncate_split(&p, &mut o, 36, 32); } t[1].push(t0.elapsed().as_secs_f64() * 1e6 / batch as f64);
        let t0 = Instant::now(); for c in &comp { let mut o = native_ks(&pc, &mut ctx, &nat, c); rerandomise(&p, &mut ctx, &pk, &mut o); truncate_split(&p, &mut o, 36, 32); } t[2].push(t0.elapsed().as_secs_f64() * 1e6 / batch as f64);
        let t0 = Instant::now(); for c in &full { let m = dec_template_b(&p, &s, c); let _ = enc_template_b(&p, &s2, &m, &mut rng); } t[3].push(t0.elapsed().as_secs_f64() * 1e6 / batch as f64);
        let t0 = Instant::now(); for c in &comp { let mut o = GlweCiphertext::new(0u64, p.glwe_size(), p.poly(), modulus()); ctx.ext(&mut o, &qb, c); } t[4].push(t0.elapsed().as_secs_f64() * 1e6 / batch as f64);
        let t0 = Instant::now(); for _ in 0..10 { let g = enc_query_b(&p, &s, &q, &mut rng); let _ = ctx.to_fourier(&g); } t[5].push(t0.elapsed().as_secs_f64() * 1e6 / 10.0);
        let t0 = Instant::now(); for c in &full { let _ = native_ks(&p, &mut ctx, &nat15, c); } t[7].push(t0.elapsed().as_secs_f64() * 1e6 / batch as f64);
        let t0 = Instant::now(); for c in &full { let _ = owner_reencrypt_fft(&p, &mut ctx, &fk1, &fk2, c); } t[8].push(t0.elapsed().as_secs_f64() * 1e6 / batch as f64);
        let t0 = Instant::now(); for c in &full { let _ = native_ks(&pc, &mut ctx, &nat, c); } t[9].push(t0.elapsed().as_secs_f64() * 1e6 / batch as f64);
        let t0 = Instant::now(); let _k = ksk_gen_seeded(&pc, &s, &s2, 4, 9); let _p = glwe_encrypt(&p, &s2, &vec![0u64; p.n], &mut rng); t[6].push(t0.elapsed().as_secs_f64() * 1e6);
    }
    let mut f = std::fs::File::create(&a[3]).unwrap();
    writeln!(f, "op,us_median,us_min,us_max,per,rounds,batch").unwrap();
    for (i, o) in ops.iter().enumerate() {
        let (m, lo, hi) = med(t[i].clone());
        let per = if i <= 4 || i >= 7 { "ciphertext(4 templates)" } else { "call" };
        writeln!(f, "{o},{m:.2},{lo:.2},{hi:.2},{per},{rounds},{batch}").unwrap();
        println!("{o:34} {m:9.2} us per {per}");
    }
    writeln!(f, "token_bytes_seeded,{},,,bytes,,", sk.bytes()).unwrap();
    writeln!(f, "pk_bytes,{},,,bytes,,", p.glwe_bytes()).unwrap();
    writeln!(f, "query_ggsw_bytes,{},,,bytes,,", p.ggsw_bytes(p.q_level)).unwrap();
    writeln!(f, "template_bytes_compact,{:.1},,,bytes per template,,", stored_bytes_per_template(&p, 36, 32)).unwrap();
}
