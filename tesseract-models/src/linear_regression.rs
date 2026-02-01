#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use tesseract_core::{Float, Matrix, Result, TesseractError, Vector};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Ordinary Least Squares (OLS) **linear regression** for a single continuous target.
///
/// This model fits parameters `(w, b)` for the affine predictor:
///
/// ```text
/// ŷ = X w + b
/// ```
///
/// where:
/// - `X` is an `(n × d)` design matrix (n samples, d features)
/// - `w` is a `(d × 1)` weight vector
/// - `b` is a scalar bias (intercept)
///
/// # Training objective
///
/// We solve the least-squares problem:
///
/// ```text
/// minimize_w,b  ||X w + b·1 - y||²₂
/// ```
///
/// Instead of augmenting `X` with a column of ones (which allocates an `(n × (d+1))` matrix),
/// this implementation uses **centering** (a common, memory-friendly formulation):
///
/// 1. Compute means:
///    - `μ_x` (feature-wise mean, length `d`)
///    - `μ_y` (target mean, scalar)
///
/// 2. Center data:
///
/// ```text
/// X_c = X - 1 μ_xᵀ
/// y_c = y - μ_y·1
/// ```
///
/// 3. Solve the centered least-squares problem:
///
/// ```text
/// minimize_w  ||X_c w - y_c||²₂
/// ```
///
/// using **SVD (Singular Value Decomposition)**:
///
/// ```text
/// X_c = U Σ Vᵀ,   w = V Σ⁻¹ Uᵀ y_c
/// ```
///
/// 4. Recover the bias (intercept):
///
/// ```text
/// b = μ_y - μ_xᵀ w
/// ```
///
/// # Numerical notes
///
/// - SVD provides numerically stable least-squares solutions and gracefully handles
///   rank-deficient matrices by zeroing out small singular values.
/// - We solve the centered system `X_c w = y_c`, which often improves conditioning.
/// - The tolerance for singular value cutoff is set to `1e-10`.
///
/// # Stored parameters
///
/// - `weights`: `Some(w)` after a successful [`fit`](LinearRegression::fit); `None` before fitting.
/// - `bias`: scalar intercept `b`. Defaults to `0.0` until fit.
///
/// # Errors
///
/// - [`TesseractError::EmptyTrainingData`] if `X` has zero rows.
/// - [`TesseractError::ShapeMismatch`] if `y.len() != X.nrows()` or prediction shapes mismatch.
/// - [`TesseractError::SingularMatrix`] if QR cannot produce a solution (e.g., rank deficiency).
/// - [`TesseractError::NotFitted`] if [`predict`](LinearRegression::predict) is called before fit.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct LinearRegression {
    /// Intercept term `b` in `ŷ = Xw + b`.
    ///
    /// This value is meaningful only after fitting, but is stored as a scalar
    /// to avoid allocating a 1×1 matrix.
    bias: Float,

    /// Weight vector `w` of shape `(d × 1)`.
    ///
    /// `None` indicates the model is not fitted yet.
    weights: Option<Vector>,
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self {
            bias: 0.0,
            weights: None,
        }
    }
}

impl LinearRegression {
    /// Creates a new, unfitted [`LinearRegression`] model.
    ///
    /// Equivalent to [`Default::default`]. The returned model must be trained with
    /// [`fit`](LinearRegression::fit) before calling [`predict`](LinearRegression::predict).
    pub fn new() -> Self {
        Self::default()
    }

    /// Fits the model parameters `(w, b)` on training data `x` and targets `y`.
    ///
    /// # Parameters
    ///
    /// - `x`: Design matrix `X` with shape `(n × d)`.
    /// - `y`: Target vector `y` with length `n`.
    ///
    /// # Algorithm (centering + QR)
    ///
    /// 1. Compute means:
    ///
    /// ```text
    /// μ_x[j] = (1/n) Σ_i X[i,j]
    /// μ_y    = (1/n) Σ_i y[i]
    /// ```
    ///
    /// 2. Center:
    ///
    /// ```text
    /// X_c[i,j] = X[i,j] - μ_x[j]
    /// y_c[i]   = y[i]   - μ_y
    /// ```
    ///
    /// 3. Solve least squares via SVD:
    ///
    /// ```text
    /// w = argmin ||X_c w - y_c||²₂
    /// ```
    ///
    /// 4. Recover bias:
    ///
    /// ```text
    /// b = μ_y - μ_xᵀ w
    /// ```
    ///
    /// # Returns
    ///
    /// - `Ok(())` on success, storing `weights` and `bias` in the struct.
    ///
    /// # Errors
    ///
    /// - [`TesseractError::EmptyTrainingData`] if `x.nrows() == 0`.
    /// - [`TesseractError::ShapeMismatch`] if `y.len() != x.nrows()`.
    /// - [`TesseractError::SingularMatrix`] if SVD cannot solve the system.
    ///
    /// # Complexity
    ///
    /// Let `n = nrows`, `d = ncols`:
    /// - Mean computation: `O(n d)`
    /// - Centering: `O(n d)`
    /// - SVD: `O(min(n,d) * n * d)` typical case
    /// - Memory: `O(n d)` for the centered copy of `X`
    pub fn fit(&mut self, x: &Matrix, y: &Vector) -> Result<()> {
        let n = x.nrows();
        let d = x.ncols();

        if n == 0 {
            return Err(TesseractError::EmptyTrainingData);
        }
        if y.len() != n {
            return Err(TesseractError::ShapeMismatch {
                expected: format!("Expected {} targets", n),
                got: format!("Got {}", y.len()),
            });
        }

        // --- Compute means μ_x (feature means) and μ_y (target mean) ---
        #[cfg(feature = "parallel")]
        let mu_x: Vector = {
            let sums: Vec<Float> = (0..d)
                .into_par_iter()
                .map(|j| x.column(j).sum())
                .collect();
            Vector::from_iterator(d, sums.into_iter().map(|s| s / n as Float))
        };

        #[cfg(not(feature = "parallel"))]
        let mu_x: Vector = {
            let mut means = Vector::zeros(d);
            for j in 0..d {
                means[j] = x.column(j).sum() / n as Float;
            }
            means
        };
        let mu_y = y.sum() / n as Float;

        // --- Center X and y: X_c = X - 1 μ_xᵀ, y_c = y - μ_y·1 ---
        let mut x_c = x.clone();
        for j in 0..d {
            let m = mu_x[j];
            x_c.column_mut(j).add_scalar_mut(-m);
        }

        let mut y_c = y.clone();
        y_c.add_scalar_mut(-mu_y);

        // --- Solve least squares: w = argmin ||X_c w - y_c||² via SVD ---
        // For overdetermined systems (n >= d), use SVD-based least squares solution
        // which is numerically stable and handles rank-deficient matrices
        let svd = x_c.clone().svd(true, true);
        let w = svd
            .solve(&y_c, 1e-10)
            .map_err(|_| TesseractError::SingularMatrix)?;

        // --- Recover intercept: b = μ_y - μ_xᵀ w ---
        let b = mu_y - mu_x.dot(&w);

        self.weights = Some(w);
        self.bias = b;
        Ok(())
    }

    /// Predicts targets for input matrix `x` using the learned parameters.
    ///
    /// Computes:
    ///
    /// ```text
    /// ŷ = X w + b
    /// ```
    ///
    /// # Parameters
    ///
    /// - `x`: Input matrix `X` with shape `(n × d)`.
    ///
    /// # Returns
    ///
    /// A vector of predictions of length `n`.
    ///
    /// # Errors
    ///
    /// - [`TesseractError::NotFitted`] if the model has not been trained.
    /// - [`TesseractError::ShapeMismatch`] if `x.ncols() != w.nrows()`.
    ///
    /// # Complexity
    ///
    /// - Time: `O(n d)` for matrix-vector multiplication.
    /// - Space: `O(n)` for the output vector.
    pub fn predict(&self, x: &Matrix) -> Result<Vector> {
        let w = self.weights.as_ref().ok_or(TesseractError::NotFitted)?;

        if x.ncols() != w.nrows() {
            return Err(TesseractError::ShapeMismatch {
                expected: format!("Expected {} features", w.nrows()),
                got: format!("Got {}", x.ncols()),
            });
        }

        // ŷ = Xw + b
        let mut preds = x * w; // (n×d)*(d×1) -> (n×1)
        preds.add_scalar_mut(self.bias); // broadcast scalar bias across all rows
        Ok(preds)
    }

    /// Returns the learned weights.
    ///
    /// # Returns
    ///
    /// Reference to weight vector, or `None` if not fitted.
    pub fn weights(&self) -> Option<&Vector> {
        self.weights.as_ref()
    }

    /// Returns the learned bias (intercept).
    pub fn bias(&self) -> Float {
        self.bias
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix_from_vec(data: Vec<Vec<f32>>) -> Matrix {
        let rows = data.len();
        let cols = data[0].len();
        Matrix::from_fn(rows, cols, |i, j| data[i][j])
    }

    fn vector_from_slice(data: &[f32]) -> Vector {
        Vector::from_iterator(data.len(), data.iter().copied())
    }

    #[test]
    fn test_linear_regression_new() {
        let lr = LinearRegression::new();
        assert!(lr.weights().is_none());
        assert_eq!(lr.bias(), 0.0);
    }

    #[test]
    fn test_linear_regression_empty_data() {
        let mut lr = LinearRegression::new();
        let x = Matrix::zeros(0, 2);
        let y = Vector::zeros(0);
        let result = lr.fit(&x, &y);
        assert!(matches!(result, Err(TesseractError::EmptyTrainingData)));
    }

    #[test]
    fn test_linear_regression_shape_mismatch() {
        let mut lr = LinearRegression::new();
        let x = matrix_from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let y = vector_from_slice(&[1.0]); // wrong length
        let result = lr.fit(&x, &y);
        assert!(matches!(result, Err(TesseractError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_linear_regression_not_fitted() {
        let lr = LinearRegression::new();
        let x = matrix_from_vec(vec![vec![1.0, 2.0]]);
        let result = lr.predict(&x);
        assert!(matches!(result, Err(TesseractError::NotFitted)));
    }

    #[test]
    fn test_linear_regression_simple_fit() {
        let mut lr = LinearRegression::new();
        // y = 2x + 3
        let x = matrix_from_vec(vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]]);
        let y = vector_from_slice(&[5.0, 7.0, 9.0, 11.0]);
        
        lr.fit(&x, &y).unwrap();

        // Check weights and bias are approximately correct
        let w = lr.weights().unwrap();
        assert!((w[0] - 2.0).abs() < 0.01);
        assert!((lr.bias() - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_linear_regression_predict() {
        let mut lr = LinearRegression::new();
        // y = 2x + 3
        let x = matrix_from_vec(vec![vec![1.0], vec![2.0], vec![3.0]]);
        let y = vector_from_slice(&[5.0, 7.0, 9.0]);
        
        lr.fit(&x, &y).unwrap();

        // Predict on new data
        let x_test = matrix_from_vec(vec![vec![5.0], vec![6.0]]);
        let y_pred = lr.predict(&x_test).unwrap();

        assert!((y_pred[0] - 13.0).abs() < 0.01);
        assert!((y_pred[1] - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_linear_regression_multivariate() {
        let mut lr = LinearRegression::new();
        // y = 2*x1 + 3*x2 + 1
        let x = matrix_from_vec(vec![
            vec![1.0, 1.0],
            vec![2.0, 1.0],
            vec![1.0, 2.0],
            vec![2.0, 2.0],
        ]);
        let y = vector_from_slice(&[6.0, 8.0, 9.0, 11.0]);
        
        lr.fit(&x, &y).unwrap();

        let w = lr.weights().unwrap();
        assert!((w[0] - 2.0).abs() < 0.01);
        assert!((w[1] - 3.0).abs() < 0.01);
        assert!((lr.bias() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_linear_regression_intercept_only() {
        let mut lr = LinearRegression::new();
        // y = 5 (constant, no x dependence)
        let x = matrix_from_vec(vec![vec![1.0], vec![2.0], vec![3.0]]);
        let y = vector_from_slice(&[5.0, 5.0, 5.0]);
        
        lr.fit(&x, &y).unwrap();

        let w = lr.weights().unwrap();
        assert!(w[0].abs() < 0.01); // Weight should be ~0
        assert!((lr.bias() - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_linear_regression_zero_intercept() {
        let mut lr = LinearRegression::new();
        // y = 2x (passes through origin)
        let x = matrix_from_vec(vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]]);
        let y = vector_from_slice(&[2.0, 4.0, 6.0, 8.0]);
        
        lr.fit(&x, &y).unwrap();

        let w = lr.weights().unwrap();
        assert!((w[0] - 2.0).abs() < 0.01);
        assert!(lr.bias().abs() < 0.01);
    }

    #[test]
    fn test_linear_regression_predict_shape_mismatch() {
        let mut lr = LinearRegression::new();
        let x = matrix_from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let y = vector_from_slice(&[1.0, 2.0]);
        lr.fit(&x, &y).unwrap();

        // Try to predict with wrong number of features
        let x_test = matrix_from_vec(vec![vec![1.0]]);
        let result = lr.predict(&x_test);
        assert!(matches!(result, Err(TesseractError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_linear_regression_perfect_fit() {
        let mut lr = LinearRegression::new();
        // Perfect linear relationship
        let x = matrix_from_vec(vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
        ]);
        let y = vector_from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);
        
        lr.fit(&x, &y).unwrap();

        // Should achieve near-perfect predictions
        let y_pred = lr.predict(&x).unwrap();
        for i in 0..y.len() {
            assert!((y_pred[i] - y[i]).abs() < 0.01);
        }
    }

    #[test]
    fn test_linear_regression_negative_values() {
        let mut lr = LinearRegression::new();
        let x = matrix_from_vec(vec![vec![-2.0], vec![-1.0], vec![0.0], vec![1.0], vec![2.0]]);
        let y = vector_from_slice(&[-4.0, -2.0, 0.0, 2.0, 4.0]);
        
        lr.fit(&x, &y).unwrap();

        let w = lr.weights().unwrap();
        assert!((w[0] - 2.0).abs() < 0.01);
        assert!(lr.bias().abs() < 0.01);
    }

    #[test]
    fn test_linear_regression_high_dimensional() {
        let mut lr = LinearRegression::new();
        // 5 features
        let x = matrix_from_vec(vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3.0, 4.0, 5.0, 6.0, 7.0],
        ]);
        let y = vector_from_slice(&[15.0, 20.0, 25.0]);
        
        lr.fit(&x, &y).unwrap();

        // Should fit without error
        assert!(lr.weights().is_some());
    }

    #[test]
    fn test_linear_regression_single_sample() {
        let mut lr = LinearRegression::new();
        let x = matrix_from_vec(vec![vec![1.0, 2.0]]);
        let y = vector_from_slice(&[5.0]);
        
        // May fail with singular matrix (underdetermined system)
        let result = lr.fit(&x, &y);
        // Either succeeds or returns SingularMatrix error
        if result.is_ok() {
            assert!(lr.weights().is_some());
        } else {
            assert!(matches!(result, Err(TesseractError::SingularMatrix)));
        }
    }

    #[test]
    fn test_linear_regression_collinear_features() {
        let mut lr = LinearRegression::new();
        // x2 = 2 * x1 (perfectly collinear)
        let x = matrix_from_vec(vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
        ]);
        let y = vector_from_slice(&[1.0, 2.0, 3.0]);
        
        // May fail with singular matrix
        let result = lr.fit(&x, &y);
        // Can either succeed or fail depending on numerical tolerance
        if result.is_err() {
            assert!(matches!(result, Err(TesseractError::SingularMatrix)));
        }
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_linear_regression_serialize_deserialize_json() {
        let mut lr = LinearRegression::new();
        let x = matrix_from_vec(vec![vec![1.0], vec![2.0], vec![3.0]]);
        let y = vector_from_slice(&[2.0, 4.0, 6.0]);
        lr.fit(&x, &y).unwrap();

        // Serialize
        let serialized = serde_json::to_string(&lr).expect("Failed to serialize");
        assert!(!serialized.is_empty());

        // Deserialize
        let deserialized: LinearRegression = serde_json::from_str(&serialized).expect("Failed to deserialize");

        // Verify predictions match
        let x_test = matrix_from_vec(vec![vec![4.0], vec![5.0]]);
        let result_original = lr.predict(&x_test).unwrap();
        let result_deserialized = deserialized.predict(&x_test).unwrap();

        for i in 0..result_original.len() {
            assert!((result_original[i] - result_deserialized[i]).abs() < 1e-6);
        }
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_linear_regression_serialize_unfitted() {
        let lr = LinearRegression::new();

        let serialized = serde_json::to_string(&lr).expect("Failed to serialize");
        let deserialized: LinearRegression = serde_json::from_str(&serialized).expect("Failed to deserialize");

        let x = matrix_from_vec(vec![vec![1.0]]);
        assert!(matches!(lr.predict(&x), Err(TesseractError::NotFitted)));
        assert!(matches!(deserialized.predict(&x), Err(TesseractError::NotFitted)));
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_linear_regression_roundtrip_preserves_state() {
        let mut lr = LinearRegression::new();
        let x = matrix_from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let y = vector_from_slice(&[5.0, 11.0]);
        lr.fit(&x, &y).unwrap();

        let serialized = serde_json::to_string(&lr).unwrap();
        let deserialized: LinearRegression = serde_json::from_str(&serialized).unwrap();

        // Weights and bias should be preserved
        let w1 = lr.weights().unwrap();
        let w2 = deserialized.weights().unwrap();
        assert_eq!(w1.len(), w2.len());
        for i in 0..w1.len() {
            assert!((w1[i] - w2[i]).abs() < 1e-6);
        }
        assert!((lr.bias() - deserialized.bias()).abs() < 1e-6);
    }

    #[test]
    fn test_linear_regression_large_dataset() {
        let mut lr = LinearRegression::new();
        // Generate larger dataset: y = 3x + 7
        let n = 100;
        let mut x_data = vec![];
        let mut y_data = vec![];
        
        for i in 0..n {
            let xi = i as f32;
            x_data.push(vec![xi]);
            y_data.push(3.0 * xi + 7.0);
        }
        
        let x = matrix_from_vec(x_data);
        let y = vector_from_slice(&y_data);
        
        lr.fit(&x, &y).unwrap();

        let w = lr.weights().unwrap();
        assert!((w[0] - 3.0).abs() < 0.01);
        assert!((lr.bias() - 7.0).abs() < 0.01);
    }

    #[test]
    fn test_linear_regression_with_noise() {
        let mut lr = LinearRegression::new();
        // y = 2x + 3 with small noise
        let x = matrix_from_vec(vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
        ]);
        let y = vector_from_slice(&[5.1, 6.9, 9.2, 10.8, 13.1]);
        
        lr.fit(&x, &y).unwrap();

        // Should still recover approximately correct parameters
        let w = lr.weights().unwrap();
        assert!((w[0] - 2.0).abs() < 0.5); // More tolerance due to noise
        assert!((lr.bias() - 3.0).abs() < 0.5);
    }
}
