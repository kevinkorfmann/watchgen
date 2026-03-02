use anyhow::Result;
use clap::Parser;
use serde::Serialize;
use std::fs;

use lshmm_rs::copying_model;
use lshmm_rs::haploid;

#[derive(Parser)]
#[command(name = "lshmm-rs", about = "Li-Stephens HMM haplotype copying model")]
struct Cli {
    /// Reference panel file (TSV: rows=sites, cols=haplotypes, values 0/1)
    #[arg(long)]
    panel: String,

    /// Query haplotype file (one allele per line, 0/1)
    #[arg(long)]
    query: String,

    /// Output JSON file
    #[arg(long, default_value = "lshmm_result.json")]
    output: String,

    /// Per-site recombination rate (uniform)
    #[arg(long, default_value_t = 0.04)]
    rho: f64,

    /// Mutation probability (0 = auto-estimate from n)
    #[arg(long, default_value_t = 0.0)]
    mu: f64,
}

#[derive(Serialize)]
struct LshmmOutput {
    n_haplotypes: usize,
    n_sites: usize,
    mu: f64,
    forward_ll: f64,
    viterbi_ll: f64,
    viterbi_path: Vec<usize>,
    posterior_path: Vec<usize>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Read reference panel
    let panel_text = fs::read_to_string(&cli.panel)?;
    let mut h: Vec<Vec<i8>> = Vec::new();
    for line in panel_text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let row: Vec<i8> = line
            .split_whitespace()
            .map(|x| x.parse::<i8>().unwrap_or(0))
            .collect();
        h.push(row);
    }
    let m = h.len();
    let n = if m > 0 { h[0].len() } else { 0 };

    // Read query
    let query_text = fs::read_to_string(&cli.query)?;
    let s: Vec<i8> = query_text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse::<i8>().unwrap_or(0))
        .collect();

    assert_eq!(s.len(), m, "Query length must match panel sites");

    let mu = if cli.mu > 0.0 {
        cli.mu
    } else {
        copying_model::estimate_mutation_probability(n)
    };

    let num_alleles = vec![2usize; m];
    let e = copying_model::emission_matrix_haploid(mu, m, &num_alleles);

    let mut r = vec![cli.rho; m];
    r[0] = 0.0;
    let r = copying_model::compute_recombination_probs(&r, n);

    // Forward-backward
    let fwd = haploid::forward(n, m, &h, &s, &e, &r);
    let bwd = haploid::backward(n, m, &h, &s, &e, &fwd.c, &r);
    let (_, posterior_path) = haploid::posterior_decoding(&fwd.f, &bwd);

    // Viterbi
    let (v, p, viterbi_ll) = haploid::viterbi(n, m, &h, &s, &e, &r);
    let viterbi_path = haploid::viterbi_traceback(m, &v, &p);

    let output = LshmmOutput {
        n_haplotypes: n,
        n_sites: m,
        mu,
        forward_ll: fwd.ll,
        viterbi_ll,
        viterbi_path,
        posterior_path,
    };

    let json = serde_json::to_string_pretty(&output)?;
    fs::write(&cli.output, &json)?;
    eprintln!("Output written to {}", cli.output);
    eprintln!(
        "n={n}, m={m}, mu={mu:.6}, forward_ll={:.4}, viterbi_ll={:.4}",
        fwd.ll, viterbi_ll
    );

    Ok(())
}
