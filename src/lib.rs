pub mod core;
pub mod datasets;
pub mod io;
pub mod metrics;
pub mod models;
pub mod optimizers;
pub mod preprocessors;

// Re-export public API
pub use core::{Float, Matrix, Result, TesseractError, Vector};
pub use datasets::*;
pub use io::*;
pub use metrics::*;
pub use models::*;
pub use optimizers::*;
pub use preprocessors::*;
