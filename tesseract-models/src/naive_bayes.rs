use tesseract_core::{Predictions, Result};

pub struct NaiveBayes {}

impl Default for NaiveBayes {
    fn default() -> Self {
        Self {}
    }
}

impl NaiveBayes {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fit() {}
    pub fn predict() -> Result<Predictions> {
        let predictions = vec![0usize; 5];
        Ok(predictions)
    }
}
