use tesseract_core::{Matrix, Result, TesseractError};

pub struct LinearRegression {
    bias: Option<Matrix>,
    weights: Option<Matrix>,
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self {
            bias: None,
            weights: None,
        }
    }
}

impl LinearRegression {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fit(&self) {}

    pub fn predict(&self, x: &Matrix) -> Result<Matrix> {
        let w = self.weights.as_ref().ok_or(TesseractError::NotFitted)?;

        // Shape check: x (n×d), w (d×k)
        if x.ncols() != w.nrows() {
            return Err(TesseractError::ShapeMismatch {
                expected: format!("Expected {} features", w.nrows()),
                got: format!("Got {} features", x.ncols()),
            });
        }

        // Matrix multiply: (n×d) * (d×k) = (n×k)
        let mut preds = x * w;

        // Optional bias add (broadcast row vector)
        if let Some(b) = self.bias.as_ref() {
            if b.nrows() != 1 || b.ncols() != preds.ncols() {
                return Err(TesseractError::ShapeMismatch {
                    expected: format!("Bias shape (1, {})", preds.ncols()),
                    got: format!("Got ({}, {})", b.nrows(), b.ncols()),
                });
            }
            let brow = b.row(0);
            for mut row in preds.row_iter_mut() {
                row += brow;
            }
        }

        Ok(preds)
    }
}
