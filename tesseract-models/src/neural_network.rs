use tesseract_core::{Predictions, Result};

pub struct NeuralNetwork {}

impl Default for NeuralNetwork {
    fn default() -> Self {
        Self {}
    }
}

impl NeuralNetwork {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fit() {}
    pub fn predict() -> Result<Predictions> {
        let predictions = vec![0usize; 5];
        Ok(predictions)
    }
}
