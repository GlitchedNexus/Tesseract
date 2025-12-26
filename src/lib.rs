//! Lattice is minimal machine learning library
//! with a simple API

pub mod core;
pub mod encoders;
pub mod io;
pub mod metrics;
pub mod models;
pub mod optimizers;

// Re-export public API
pub use core::*;
pub use encoders::*;
pub use io::*;
pub use metrics::*;
pub use models::*;
pub use optimizers::*;
