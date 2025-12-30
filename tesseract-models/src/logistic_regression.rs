use tesseract_core::{Matrix, Result};

pub struct LogisticRegression {
    bias: Option<Matrix>,
    weights: Option<Matrix>,
}

impl Default for LogisticRegression {
    fn default() -> Self {
        Self {
            bias: None,
            weights: None,
        }
    }
}

impl LogisticRegression {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fit(&self) {}

    pub fn predict(&self, x: &Matrix) {}
}
