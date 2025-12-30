use tesseract_core::{Float, Matrix, Result, TesseractError, Vector};

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
/// using **QR decomposition**:
///
/// ```text
/// X_c = Q R,   solve R w = Qᵀ y_c
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
/// - QR is typically more numerically stable than normal equations `XᵀX w = Xᵀy`,
///   because it avoids forming `XᵀX` (which can square the condition number).
/// - We solve the centered system `X_c w = y_c`, which often improves conditioning.
/// - If the system is rank-deficient (e.g., perfectly collinear features), the solver may fail.
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
    /// 3. Solve least squares via QR:
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
    /// - [`TesseractError::SingularMatrix`] if QR cannot solve the system.
    ///
    /// # Complexity
    ///
    /// Let `n = nrows`, `d = ncols`:
    /// - Mean computation: `O(n d)`
    /// - Centering: `O(n d)`
    /// - QR solve (typical): `O(n d²)` for `n ≥ d`
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
        let mut mu_x = Vector::zeros(d);
        for j in 0..d {
            mu_x[j] = x.column(j).sum() / n as Float;
        }
        let mu_y = y.sum() / n as Float;

        // --- Center X and y: X_c = X - 1 μ_xᵀ, y_c = y - μ_y·1 ---
        let mut x_c = x.clone();
        for j in 0..d {
            let m = mu_x[j];
            x_c.column_mut(j).add_scalar_mut(-m);
        }

        let mut y_c = y.clone();
        y_c.add_scalar_mut(-mu_y);

        // --- Solve least squares: w = argmin ||X_c w - y_c||² via QR ---
        let w = x_c.qr().solve(&y_c).ok_or(TesseractError::SingularMatrix)?;

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
}
