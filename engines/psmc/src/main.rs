use std::fs;
use std::io::BufReader;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use serde::Serialize;

use psmc_rs::decode::{get_t_boundaries, plot_psmc_history, scale_psmc_output, ScaledOutput};
use psmc_rs::inference::{psmc_inference, IterationResult, PsmcConfig};
use psmc_rs::psmcfa::read_psmcfa;

#[derive(Parser, Debug)]
#[command(name = "psmc-rs", about = "PSMC inference in Rust")]
struct Cli {
    /// Input PSMCFA file
    #[arg(short, long)]
    input: PathBuf,

    /// Output JSON file (default: stdout)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Number of EM iterations
    #[arg(long, default_value_t = 25)]
    n_iters: usize,

    /// Maximum coalescent time
    #[arg(long, default_value_t = 15.0)]
    t_max: f64,

    /// Theta / rho ratio
    #[arg(long, default_value_t = 5.0)]
    ratio: f64,

    /// PSMC pattern string
    #[arg(long, default_value = "4+25*2+4+6")]
    pattern: String,

    /// Per-base per-generation mutation rate
    #[arg(long, default_value_t = 1.25e-8)]
    mu: f64,

    /// Bin size in base pairs
    #[arg(long, default_value_t = 100)]
    bin_size: u64,

    /// Generation time in years
    #[arg(long, default_value_t = 25.0)]
    generation_time: f64,

    /// Alpha parameter for time discretization
    #[arg(long, default_value_t = 0.1)]
    alpha: f64,
}

#[derive(Serialize)]
struct PsmcOutput {
    config: PsmcConfig,
    input_stats: InputStats,
    iterations: Vec<IterationResult>,
    final_params: FinalParams,
    scaled_output: ScaledOutput,
    plot_data: PlotData,
}

#[derive(Serialize)]
struct InputStats {
    n_records: usize,
    total_bins: usize,
    n_het: usize,
    n_hom: usize,
    n_missing: usize,
    het_fraction: f64,
}

#[derive(Serialize)]
struct FinalParams {
    theta: f64,
    rho: f64,
    lambdas: Vec<f64>,
    t_boundaries: Vec<f64>,
}

#[derive(Serialize)]
struct PlotData {
    x_years: Vec<f64>,
    y_pop_sizes: Vec<f64>,
}

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    // Read input
    let file = fs::File::open(&cli.input)
        .with_context(|| format!("failed to open {}", cli.input.display()))?;
    let reader = BufReader::new(file);
    let records = read_psmcfa(reader).context("failed to parse PSMCFA")?;

    // Concatenate all records
    let mut seq: Vec<u8> = Vec::new();
    for rec in &records {
        seq.extend_from_slice(&rec.seq);
    }

    let n_het = seq.iter().filter(|&&v| v == 1).count();
    let n_hom = seq.iter().filter(|&&v| v == 0).count();
    let n_missing = seq.iter().filter(|&&v| v >= 2).count();
    let n_valid = n_het + n_hom;
    let het_fraction = if n_valid > 0 {
        n_het as f64 / n_valid as f64
    } else {
        0.0
    };

    let input_stats = InputStats {
        n_records: records.len(),
        total_bins: seq.len(),
        n_het,
        n_hom,
        n_missing,
        het_fraction,
    };

    eprintln!(
        "Loaded {} records, {} total bins ({} het, {} hom, {} missing)",
        records.len(),
        seq.len(),
        n_het,
        n_hom,
        n_missing
    );

    // Configure and run inference
    let config = PsmcConfig {
        n: 63,
        t_max: cli.t_max,
        theta_rho_ratio: cli.ratio,
        pattern: cli.pattern.clone(),
        n_iters: cli.n_iters,
        alpha_param: cli.alpha,
    };

    let results = psmc_inference(&seq, &config)?;

    // Extract final parameters from last iteration
    let last = results.last().expect("no iterations completed");
    let final_theta = last.theta;
    let final_rho = last.rho;
    let final_lambdas = last.lambdas.clone();

    let t_boundaries = get_t_boundaries(config.n, config.t_max, config.alpha_param);

    let scaled = scale_psmc_output(
        final_theta,
        &final_lambdas,
        &t_boundaries,
        cli.mu,
        cli.bin_size as f64,
        cli.generation_time,
    );

    let (x_years, y_pop_sizes) = plot_psmc_history(
        final_theta,
        &final_lambdas,
        &t_boundaries,
        cli.mu,
        cli.bin_size as f64,
        cli.generation_time,
    );

    let output = PsmcOutput {
        config,
        input_stats,
        iterations: results,
        final_params: FinalParams {
            theta: final_theta,
            rho: final_rho,
            lambdas: final_lambdas,
            t_boundaries: t_boundaries.clone(),
        },
        scaled_output: scaled,
        plot_data: PlotData {
            x_years,
            y_pop_sizes,
        },
    };

    let json = serde_json::to_string_pretty(&output).context("failed to serialize JSON")?;

    if let Some(out_path) = &cli.output {
        fs::write(out_path, &json)
            .with_context(|| format!("failed to write {}", out_path.display()))?;
        eprintln!("Output written to {}", out_path.display());
    } else {
        println!("{json}");
    }

    Ok(())
}
