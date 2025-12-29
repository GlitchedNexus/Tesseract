pub mod error;
pub mod impurity;
pub mod loss;
pub mod types;
pub mod utils;

pub use types::{Float, Label, Matrix, Predictions, Scalar, Vector};

pub use error::{Result, TesseractError};

pub use impurity::gini_from_counts;
