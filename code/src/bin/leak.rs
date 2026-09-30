//! Key-leakage experiment for post-compromise security.
//! An adversary holding s_{i+1} and snapshots of the same stored ciphertexts at epochs i and i+1 observes
//!   h = phase_{s_{i+1}}(C') - m = e_C + sum_{l,i} d_{l,i}(a) E_{l,i} + e_rr + e_rd,
//! i.e. a hint on the error e_C of the epoch-i sample (a, b = <a,s_i> + m + e_C).
//! We measure every term, give the adversary the token noise E for free, and report the residual error
//! of the best linear estimate of e_C from the hint (the effective error of the hinted sample on s_i).
//! usage: RF_PARAMS=p128 leak <n_ciphertexts> <out.csv>
use rotaface::*;
use std::io::Write;
use tfhe::core_crypto::prelude::*;

fn signed(v: u64) -> f64 { v as i64 as f64 }
fn mean(v: &[f64]) -> f64 { v.iter().sum::<f64>() / v.len() as f64 }
fn var(v: &[f64]) -> f64 { let m = mean(v); v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64 }
fn cov(a: &[f64], b: &[f64]) -> f64 { let (ma, mb) = (mean(a), mean(b)); a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum::<f64>() / a.len() as f64 }
fn kurt(v: &[f64]) -> f64 { let m = mean(v); let s2 = var(v); v.iter().map(|x| (x - m).powi(4)).sum::<f64>() / v.len() as f64 / (s2 * s2) - 3.0 }
fn l2(x: f64) -> f64 { x.sqrt().log2() }

/// Signed gadget digits of the top bl*lv bits, identical to native_ks.
fn digits(a: &[u64], bl: usize, lv: usize) -> Vec<Vec<u64>> {
    let n = a.len(); let tot = bl * lv; let bmask = (1u64 << bl) - 1; let half = 1u64 << (bl - 1);
    let sh = 64 - tot; let rnd = 1u64 << (63 - tot);
    let mut xs: Vec<u64> = a.iter().map(|v| v.wrapping_add(rnd) >> sh).collect();
    let mut dig = vec![vec![0u64; n]; lv];
    for l in (0..lv).rev() {
        for (dc, xc) in dig[l].iter_mut().zip(xs.iter_mut()) {
            let x = *xc; let di = x & bmask; let carry = (di >= half) as u64;
            *dc = di.wrapping_sub(carry << bl); *xc = (x >> bl).wrapping_add(carry);
        }
    }
    dig
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let m: usize = a[1].parse().unwrap(); let out = &a[2];
    let p = params_from_env();
    let (bl, lv) = (4usize, 9usize);
    let mut pc = p; pc.r_base_log = bl; pc.r_level = lv;
    let (k, n) = (p.k, p.n);
    let mut rng = Rng::new(); let mut ctx = Ctx::new(&p);
    let s = keygen(&p, &mut rng); let s2 = keygen(&p, &mut rng);
    let sk = ksk_gen_seeded(&pc, &s, &s2, bl, lv);
    let nat = NativeKsk::expand(&pc, &mut ctx, &sk);
    let pk = pk_gen(&p, &mut ctx, &s2, &mut rng);
    // token noise E_{l,i} = phase_{s2}(K_{l,i}) + beta_l * s_i  (given to the adversary for free)
    let mut rows = GlweCiphertextList::new(0u64, p.glwe_size(), p.poly(), GlweCiphertextCount(lv * k), modulus());
    decompress_seeded_glwe_ciphertext_list::<_, _, _, ActivatedRandomGenerator>(&mut rows, &sk.list);
    let so = key_polys(&p, &s);
    let mut e_tok: Vec<Vec<u64>> = Vec::new();
    for (r, row) in rows.iter().enumerate() {
        let (l, i) = (r / k, r % k);
        let beta = 1u64 << (64 - bl * (l + 1));
        let ph = glwe_phase(&p, &s2, &row);
        e_tok.push(ph.iter().zip(&so[i]).map(|(x, si)| x.wrapping_add(si.wrapping_mul(beta))).collect());
    }
    let etok_f: Vec<f64> = e_tok.iter().flatten().map(|&v| signed(v)).collect();
    let mut ur = rand::thread_rng();
    let (mut ec, mut de, mut rr, mut rd, mut hint) = (vec![], vec![], vec![], vec![], vec![]);
    for _ in 0..m {
        let x: Vec<i64> = (0..n).map(|_| rand::Rng::gen_range(&mut ur, -600i64..=600)).collect();
        let msg = smul(&wrap(&x), p.delta());
        let mut c = glwe_encrypt(&p, &s, &msg, &mut rng);
        truncate_split(&p, &mut c, 36, 32); // stored epoch-i ciphertext
        let e_c: Vec<u64> = glwe_phase(&p, &s, &c).iter().zip(&msg).map(|(ph, mm)| ph.wrapping_sub(*mm)).collect();
        // sum_{l,i} d_{l,i}(a_i) * E_{l,i}
        let d = c.as_ref();
        let mut sde = vec![0u64; n];
        for i in 0..k {
            let dig = digits(&d[i * n..(i + 1) * n], bl, lv);
            for l in 0..lv { let t = pmul(&dig[l], &e_tok[l * k + i]); for (z, y) in sde.iter_mut().zip(&t) { *z = z.wrapping_add(*y); } }
        }
        let mut o = native_ks(&pc, &mut ctx, &nat, &c);
        rerandomise(&p, &mut ctx, &pk, &mut o);
        let ph_pre: Vec<u64> = glwe_phase(&p, &s2, &o);
        truncate_split(&p, &mut o, 36, 32); // stored epoch-(i+1) ciphertext
        let ph_post: Vec<u64> = glwe_phase(&p, &s2, &o);
        for j in 0..n {
            let h = ph_post[j].wrapping_sub(msg[j]);
            let r_rr = ph_pre[j].wrapping_sub(msg[j]).wrapping_sub(e_c[j]).wrapping_sub(sde[j]);
            let r_rd = ph_post[j].wrapping_sub(ph_pre[j]);
            ec.push(signed(e_c[j])); de.push(signed(sde[j])); rr.push(signed(r_rr)); rd.push(signed(r_rd));
            hint.push(signed(h.wrapping_sub(sde[j]))); // adversary knows E, hence sde
        }
    }
    // best linear estimate of e_C from the hint and its residual (effective error of the hinted sample)
    let lam = cov(&ec, &hint) / var(&hint);
    let resid: Vec<f64> = ec.iter().zip(&hint).map(|(e, h)| e - lam * h).collect();
    let sigma_fresh = p.std_rel * 2f64.powi(64);
    // cross-ciphertext correlation of the residual hint noise at the same coefficient (independence given E)
    let noise: Vec<f64> = rr.iter().zip(&rd).map(|(a, b)| a + b).collect();
    let (x0, x1): (Vec<f64>, Vec<f64>) = (0..(m - 1) * n).map(|t| (noise[t], noise[t + n])).unzip();
    let corr_cross = cov(&x0, &x1) / (var(&x0) * var(&x1)).sqrt();
    let corr_ec_noise = cov(&ec, &noise) / (var(&ec) * var(&noise)).sqrt();
    let mut f = std::fs::File::create(out).unwrap();
    writeln!(f, "quantity,log2_std_or_value").unwrap();
    let rows_out = [
        ("fresh_sigma", sigma_fresh.log2()), ("token_noise_E", l2(var(&etok_f))),
        ("e_C_stored_epoch_i", l2(var(&ec))), ("sum_dE", l2(var(&de))),
        ("e_rr_rerand", l2(var(&rr))), ("e_rd_rounding", l2(var(&rd))),
        ("hint_noise_rr_plus_rd", l2(var(&noise))), ("effective_error_hinted_sample", l2(var(&resid))),
        ("lambda", lam), ("corr_eC_hintnoise", corr_ec_noise), ("corr_hintnoise_across_ciphertexts", corr_cross),
        ("excess_kurtosis_e_rr", kurt(&rr)), ("excess_kurtosis_hintnoise", kurt(&noise)), ("samples", (m * n) as f64),
    ];
    for (q, v) in rows_out { writeln!(f, "{q},{v:.4}").unwrap(); println!("{q:36} {v:.4}"); }
}
