use thiserror::Error;

#[derive(Debug, Error)]
pub enum PsmcError {
    #[error("invalid pattern: {0}")]
    InvalidPattern(String),

    #[error("pattern gives {got} intervals, but need {expected}")]
    PatternMismatch { expected: usize, got: usize },

    #[error("invalid PSMCFA: {0}")]
    InvalidPsmcfa(String),

    #[error("numerical error: {0}")]
    Numerical(String),

    #[error("empty sequence")]
    EmptySequence,

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
