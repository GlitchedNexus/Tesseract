pub(crate) mod core;
pub mod datasets;
pub mod io;
pub mod metrics;
pub mod models;
pub mod optimizers;
pub mod preprocessors;

// Re-export public API
pub use core::{Float, Label, Matrix, Predictions, Result, Scalar, TesseractError, Vector};
pub use datasets::*;
pub use io::*;
pub use metrics::*;
pub use models::*;
pub use optimizers::*;
pub use preprocessors::*;
