# Tesseract

**Tesseract** is a modular, multi-modal machine learning library written in Rust.
It focuses on correctness, performance, and a clean developer experience while supporting classical machine learning workflows over numerical (and later text and image) data.

The project is organized as a Cargo workspace with small, focused crates and a thin public facade.

---

## Design Principles

1. **Minimal dependencies and resource usage**
   Each subsystem lives in its own crate with narrowly scoped dependencies. Heavy dependencies are optional and feature-gated.

2. **Clear separation of concerns**
   Core types, models, metrics, optimizers, and I/O are isolated to improve maintainability and compile times.

3. **Correctness first**
   Strong typing, explicit error handling, and predictable APIs are prioritized over implicit behavior.

4. **Practical documentation**
   Public APIs are documented with usage examples and clear error semantics.

---

## Workspace Structure

```
tesseract/
├── tesseract              # Public facade crate
├── tesseract-core         # Core types, errors, numeric aliases
├── tesseract-io           # Data loading (CSV, etc.)
├── tesseract-metrics      # Evaluation metrics
├── tesseract-models       # ML models
├── tesseract-optimizers   # Gradient-based optimizers
├── tesseract-preprocessors# Data preprocessing utilities
└── Cargo.toml             # Workspace manifest
```

The `tesseract` crate re-exports the public API, while internal crates depend directly on `tesseract-core`.

---

## Core Concepts

* **Matrix / Vector backend**:
  Uses `nalgebra` for 2D matrices and vectors (`DMatrix`, `DVector`).

* **Numeric types**:

  * `Float` (`f32`) for model parameters and data
  * `Scalar` (`f64`) for aggregated statistics and metrics

* **Errors**:
  All fallible operations return a unified `Result<T, TesseractError>`.

---

## Model Support

### Implemented

* k-Nearest Neighbors (k-NN)
* Decision Stump (classification)
* Linear Regression

### Planned / In Progress

* Naive Bayes
* Decision Tree
* Random Tree / Random Forest
* Logistic Regression
* k-Means Clustering
* Neural Networks

The initial focus is on **classical machine learning models operating on 2D feature matrices**. Higher-dimensional tensor models (e.g., CNNs) are planned for later phases.

---

## Example

```rust
use tesseract::{Matrix, LinearRegression};

let x = Matrix::from_row_slice(
    3, 2,
    &[
        1.0, 2.0,
        2.0, 3.0,
        3.0, 4.0,
    ],
);

let y = vec![3.0, 5.0, 7.0];

let mut model = LinearRegression::new();
model.fit(&x, &y).unwrap();

let preds = model.predict(&x).unwrap();
```

---

## Feature Flags

Optional dependencies are feature-gated:

* `io-csv` — CSV loading support
* `rng` — Randomized algorithms (e.g., k-NN tie breaking, initialization)

By default, common features are enabled. You can opt out with:

```bash
cargo build --no-default-features
```

---

## Status

Tesseract is under active development and should be considered **pre-stable**.
APIs may evolve as models and abstractions mature.

---

## License

MIT License.
