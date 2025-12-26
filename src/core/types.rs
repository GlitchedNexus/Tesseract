pub type Float = f32;
pub type Matrix = nalgebra::DMatrix<Float>;
pub type Vector = nalgebra::DVector<Float>;
pub type Result<T> = std::result::Result<T, TesseractError>;

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
