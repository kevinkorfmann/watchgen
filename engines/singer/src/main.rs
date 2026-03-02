use anyhow::Result;
use clap::Parser;
use serde::Serialize;
use std::fs;

use singer_rs::branch_sampling;

#[derive(Parser)]
#[command(
    name = "singer-rs",
    about = "SINGER: Sampling and Inference of Genealogies with Recombination"
)]
struct Cli {
    /// Number of haploid samples
    #[arg(long, default_value_t = 10)]
    n_samples: usize,

    /// Number of time grid points for branch partitioning
    #[arg(long, default_value_t = 20)]
    n_times: usize,

    /// Scaled mutation rate (theta = 4*Ne*mu)
    #[arg(long, default_value_t = 0.001)]
    theta: f64,

    /// Scaled recombination rate (rho = 4*Ne*r per site)
    #[arg(long, default_value_t = 0.0004)]
    rho: f64,

    /// Maximum time in coalescent units
    #[arg(long, default_value_t = 10.0)]
    t_max: f64,

    /// Output JSON file
    #[arg(long, default_value = "singer_result.json")]
    output: String,
}

#[derive(Serialize)]
struct SingerOutput {
    n_samples: usize,
    n_times: usize,
    theta: f64,
    rho: f64,
    t_max: f64,
    branch_time_grid: Vec<f64>,
    joining_probs: Vec<f64>,
    f_bar_values: Vec<f64>,
    lambda_values: Vec<f64>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    eprintln!(
        "SINGER: n={}, t_max={}, theta={}, rho={}",
        cli.n_samples, cli.t_max, cli.theta, cli.rho
    );

    // Compute branch time grid
    let dt = cli.t_max / cli.n_times as f64;
    let time_grid: Vec<f64> = (0..=cli.n_times).map(|i| i as f64 * dt).collect();

    // Compute lambda, F_bar, and joining probabilities at grid points
    let lambda_values: Vec<f64> = time_grid
        .iter()
        .map(|&t| branch_sampling::lambda_approx(t, cli.n_samples))
        .collect();

    let f_bar_values: Vec<f64> = time_grid
        .iter()
        .map(|&t| branch_sampling::f_bar_approx(t, cli.n_samples))
        .collect();

    let joining_probs: Vec<f64> = (0..cli.n_times)
        .map(|i| branch_sampling::joining_prob_approx(time_grid[i], time_grid[i + 1], cli.n_samples))
        .collect();

    let total_prob: f64 = joining_probs.iter().sum();
    eprintln!("Total joining probability over grid: {total_prob:.6}");
    eprintln!(
        "Lambda range: [{:.3}, {:.3}]",
        lambda_values.last().unwrap(),
        lambda_values[0]
    );

    let output = SingerOutput {
        n_samples: cli.n_samples,
        n_times: cli.n_times,
        theta: cli.theta,
        rho: cli.rho,
        t_max: cli.t_max,
        branch_time_grid: time_grid,
        joining_probs,
        f_bar_values,
        lambda_values,
    };

    let json = serde_json::to_string_pretty(&output)?;
    fs::write(&cli.output, &json)?;
    eprintln!("Output written to {}", cli.output);

    Ok(())
}
