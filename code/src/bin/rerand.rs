//! Re-randomised rotation: correctness, extra noise, extra cost, and unlinkability sanity check.
use rotaface::*;
use std::time::Instant;
fn main() {
    let p = params_from_env();
    let emb: Vec<Vec<f32>> = load_f32(&data_path(), 512).into_iter().filter(|v| v.iter().map(|x| x * x).sum::<f32>() > 0.25).collect();
    let mut rng = Rng::new(); let mut ctx = Ctx::new(&p);
    let s = keygen(&p, &mut rng); let s2 = keygen(&p, &mut rng);
    let tok = Token::from_std(&mut ctx, &token_gen(&p, &s, &s2, false, &mut rng));
    let pk = pk_gen(&p, &mut ctx, &s2, &mut rng);
    let m = 200;
    let xs: Vec<Vec<i64>> = (0..m).map(|i| { let mut v: Vec<i64> = emb[i].iter().map(|&x| (x * p.scale).round() as i64).collect(); v.resize(p.n, 0); v }).collect();
    let cts: Vec<Glwe> = xs.iter().map(|x| enc_template_b(&p, &s, x, &mut rng)).collect();
    let t0 = Instant::now(); let plain: Vec<Glwe> = cts.iter().map(|c| rotate_glwe(&mut ctx, &tok, c)).collect(); let t_rot = t0.elapsed().as_secs_f64() / m as f64;
    let t0 = Instant::now(); let rr: Vec<Glwe> = cts.iter().map(|c| { let mut o = rotate_glwe(&mut ctx, &tok, c); rerandomise(&p, &mut ctx, &pk, &mut o); o }).collect(); let t_rr = t0.elapsed().as_secs_f64() / m as f64;
    let n0: Vec<f64> = plain.iter().zip(&xs).flat_map(|(c, x)| glwe_noise(&p, &s2, c, x)).collect();
    let n1: Vec<f64> = rr.iter().zip(&xs).flat_map(|(c, x)| glwe_noise(&p, &s2, c, x)).collect();
    println!("k={} N={}: rotate {:.1} us/ct, rotate+rerand {:.1} us/ct (+{:.0}%)", p.k, p.n, t_rot * 1e6, t_rr * 1e6, (t_rr / t_rot - 1.0) * 100.0);
    println!("noise log2 std: rotated {:.2}, rotated+rerand {:.2}", std(&n0).log2(), std(&n1).log2());
    // unlinkability sanity: rotating the same ciphertext twice gives different ciphertexts (deterministic w/o rerand)
    let a = rotate_glwe(&mut ctx, &tok, &cts[0]); let b = rotate_glwe(&mut ctx, &tok, &cts[0]);
    let mut c = a.clone(); rerandomise(&p, &mut ctx, &pk, &mut c); let mut d = b.clone(); rerandomise(&p, &mut ctx, &pk, &mut d);
    println!("same input -> identical output: without rerand {}, with rerand {}", a.as_ref() == b.as_ref(), c.as_ref() == d.as_ref());
}
