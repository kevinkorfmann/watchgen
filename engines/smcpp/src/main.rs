use anyhow::Result;
use clap::Parser;
use serde::Serialize;
use std::fs;

use smcpp_rs::hmm::composite_log_likelihood;
use smcpp_rs::inference::{fit_smcpp, SmcppConfig};

#[derive(Parser)]
#[command(name = "smcpp-rs", about = "SMC++ demographic inference from multiple samples")]
struct Cli {
    /// Input file: one observation sequence per line (space-separated 0/1 values)
    #[arg(long)]
    input: String,

    /// Output JSON file
    #[arg(long, default_value = "smcpp_result.json")]
    output: String,

    /// Number of time intervals
    #[arg(long, default_value_t = 10)]
    n_intervals: usize,

    /// Maximum time (coalescent units)
    #[arg(long, default_value_t = 5.0)]
    t_max: f64,

    /// Scaled mutation rate per bin
    #[arg(long, default_value_t = 0.001)]
    theta: f64,

    /// Scaled recombination rate per bin
    #[arg(long, default_value_t = 0.0002)]
    rho: f64,

    /// Maximum optimization iterations
    #[arg(long, default_value_t = 20)]
    max_iter: usize,
}

#[derive(Serialize)]
struct SmcppOutput {
    n_samples: usize,
    n_intervals: usize,
    theta: f64,
    rho: f64,
    initial_ll: f64,
    final_ll: f64,
    iterations: usize,
    lambdas: Vec<f64>,
    time_breaks: Vec<f64>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Read input
    let input_text = fs::read_to_string(&cli.input)?;
    let data: Vec<Vec<u8>> = input_text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split_whitespace()
                .map(|x| x.parse::<u8>().unwrap_or(0))
                .collect()
        })
        .collect();

    let n_samples = data.len();
    eprintln!("Loaded {n_samples} observation sequences");

    // Log-spaced time breaks: t_k = alpha * (exp(beta*k) - 1)
    // This gives finer resolution at recent times where most coalescence occurs
    let alpha = 0.1;
    let beta = (1.0 + cli.t_max / alpha).ln() / cli.n_intervals as f64;
    let time_breaks: Vec<f64> = (0..=cli.n_intervals)
        .map(|i| alpha * ((beta * i as f64).exp() - 1.0))
        .collect();

    let initial_ll = composite_log_likelihood(
        &data,
        &time_breaks,
        &vec![1.0; cli.n_intervals],
        cli.theta,
        cli.rho,
    );

    eprintln!("Initial LL: {initial_ll:.4}");

    let config = SmcppConfig {
        time_breaks: time_breaks.clone(),
        theta: cli.theta,
        rho: cli.rho,
        max_iter: cli.max_iter,
    };

    let result = fit_smcpp(&data, &config);

    eprintln!(
        "Final LL: {:.4} ({} iterations)",
        result.log_likelihood, result.iterations
    );

    let output = SmcppOutput {
        n_samples,
        n_intervals: cli.n_intervals,
        theta: cli.theta,
        rho: cli.rho,
        initial_ll,
        final_ll: result.log_likelihood,
        iterations: result.iterations,
        lambdas: result.lambdas,
        time_breaks,
    };

    let json = serde_json::to_string_pretty(&output)?;
    fs::write(&cli.output, &json)?;
    eprintln!("Output written to {}", cli.output);

    Ok(())
}
