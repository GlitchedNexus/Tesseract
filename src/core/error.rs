use std::fmt;

#[derive(Debug)]
pub enum TesseractError {
    /// IO-related failures (file not found, unreadable, etc.)
    Io(String),

    /// CSV parsing / formatting issues
    Csv(String),

    /// Shape or dimensionality mismatch
    ShapeMismatch { expected: String, got: String },

    /// Model used before calling `fit`
    NotFitted,

    /// Invalid hyperparameter or configuration
    InvalidParameter(String),

    /// Feature not enabled at compile time
    FeatureDisabled(&'static str),
}

impl fmt::Display for TesseractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for TesseractError {}

pub type Result<T> = std::result::Result<T, TesseractError>;
