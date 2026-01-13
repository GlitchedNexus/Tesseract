use std::any::Any;

use tesseract_core::{Float, Matrix, Predictions, Result, TesseractError};

enum NODE {
    LABEL(usize),
    NODE(DecisionTree),
}

/// Max depth is the maximum number of edges between NODE & root.
pub struct DecisionTree {
    max_depth: usize,
    feature_index: usize,
    feature_threshold: Float,
    left: Option<Box<NODE>>,
    right: Option<Box<NODE>>,
    isFitted: bool,
}

impl Default for DecisionTree {
    fn default() -> Self {
        Self {
            max_depth: 0,
            isFitted: false,
            feature_index: 0,
            feature_threshold: 0.0,
            left: None,
            right: None,
        }
    }
}

impl DecisionTree {
    pub fn new() {
        Self::default();
    }

    pub fn fit() {}

    pub fn predict(&self, x: Matrix, depth_left: usize) -> Result<Predictions> {
        if !self.isFitted {
            return Err(TesseractError::NotFitted);
        }

        // DFS For Prediction
        let predictions = Predictions::new();

        Ok(predictions)
    }
}
