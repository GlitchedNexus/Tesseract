# Tesseract

**Tesseract** is a modular, multi-modal machine learning library written in Rust.
It focuses on correctness, performance, and a clean developer experience while supporting classical machine learning workflows over numerical (and later text and image) data.

The project is organized as a Cargo workspace with small, focused crates and a thin public facade.

## Design principles

1. **Minimal dependencies and resource usage**  
   Each subsystem lives in its own crate with narrowly scoped dependencies. Heavy dependencies are optional and feature-gated.
2. **Clear separation of concerns**  
   Core types, models, metrics, optimizers, and I/O are isolated to improve maintainability and compile times.
3. **Correctness first**  
   Strong typing, explicit error handling, and predictable APIs are prioritized over implicit behavior.
4. **Practical documentation**  
   Public APIs are documented with usage examples and clear error semantics.

## Current state

Tesseract is pre-stable and under active development.

Most implemented code is in `tesseract-core` and `tesseract-models`; several other crates are currently scaffolds.

## Workspace crates

| Crate | Purpose | State |
| :--- | :--- | :--- |
| `tesseract` | Public facade crate that re-exports workspace APIs | Implemented |
| `tesseract-core` | Core types (`Matrix`, `Vector`, `Float`, `Label`), error types, impurity utilities | Implemented |
| `tesseract-models` | ML algorithms | Implemented |
| `tesseract-io` | I/O modules (feature-gated CSV module) | Scaffold/in progress |
| `tesseract-metrics` | Classification/regression metrics modules | Scaffold/in progress |
| `tesseract-datasets` | Dataset and split modules | Scaffold/in progress |
| `tesseract-optimizers` | Gradient-descent optimizer modules | Scaffold/in progress |
| `tesseract-preprocessors` | Encoders, imputers, scalers | Scaffold/in progress |

## Implemented algorithms (`tesseract-models`)

| Name | Task | Status | Notes |
| :--- | :--- | :--- | :--- |
| `KNN` | Supervised classification | Implemented + unit tested | Majority vote over nearest neighbors (squared Euclidean distance) |
| `DecisionStump` | Supervised classification | Implemented + unit tested | Depth-1 CART-style split using weighted Gini impurity |
| `DecisionTree` | Supervised classification | Implemented + unit tested | Recursive CART-style tree with depth/leaf split controls |
| `LinearRegression` | Supervised regression | Implemented + unit tested | OLS using centered data + SVD solve |
| `KMeans` | Unsupervised clustering | Implemented + unit tested | Lloyd's algorithm with K-Means++ initialization |
| `KMedians` | Unsupervised clustering | Implemented + unit tested | L1 distance + coordinate-wise median updates |

## Core concepts

- Linear algebra backend: `nalgebra` (`DMatrix`/`DVector` via type aliases)
- Numeric aliases from `tesseract-core`:
  - `Float = f32`
  - `Scalar = f64`
- Unified error type: `TesseractError`
- Shared result alias: `Result<T>`

## Quick example

```rust
use tesseract::{LinearRegression, Matrix, Result, Vector};

fn main() -> Result<()> {
    // y = 2x + 3
    let x = Matrix::from_row_slice(4, 1, &[1.0, 2.0, 3.0, 4.0]);
    let y = Vector::from_vec(vec![5.0, 7.0, 9.0, 11.0]);

    let mut model = LinearRegression::new();
    model.fit(&x, &y)?;

    let preds = model.predict(&x)?;
    println!("{preds:?}");

    Ok(())
}
```

## Feature flags (`tesseract` crate)

- Default features: `io-csv`, `rng`
- `io-csv` -> enables `tesseract-io/io-csv`
- `rng` -> enables randomized model paths in `tesseract-models` (required by `KMeans` and `KMedians`)
- `parallel` -> enables rayon-backed parallel code paths in `tesseract-models` and `tesseract-metrics`
- `serde` -> enables serde support in `tesseract-models` and `tesseract-io`
- `rkyv` -> forwards the `rkyv` feature to `tesseract-models`

Build without defaults:

```bash
cargo build --no-default-features
```

## License

MIT
