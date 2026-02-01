use tesseract_core::{Predictions, Result};

pub struct DecisionTree {}

impl Default for DecisionTree {
    fn default() -> Self {
        Self {}
    }
}

impl DecisionTree {
    pub fn new() {
        Self::default();
    }

    pub fn fit() {}

    pub fn predict(&self) -> Result<Predictions> {
        // DFS For Prediction
        let predictions = Predictions::new();

        Ok(predictions)
    }
}
