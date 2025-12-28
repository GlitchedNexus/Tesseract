use std::{collections::HashSet, fmt::Debug};

use tesseract_core::{Float, Matrix, Predictions, Result, TesseractError};

#[derive(Debug, Clone)]
pub struct DecisionStump {
    feature_index: usize,
    split_value: Float,
    left_class: usize,
    right_class: usize,
    fitted: bool,
}

impl Default for DecisionStump {
    fn default() -> Self {
        Self {
            feature_index: 1,
            split_value: 0.0,
            left_class: 0,
            right_class: 1,
            fitted: false,
        }
    }
}

impl DecisionStump {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fit(&mut self, x: &Matrix, y: &Predictions) -> Result<()> {
        let n = x.nrows();

        if n < 1 {
            return Err(TesseractError::EmptyTrainingData);
        }

        let d = x.ncols();

        let best: usize = 0;

        for feature in 0..d {
            let col = x.column(feature);

            for threshold in col {
                let mut left: Predictions = vec![];
                let mut right: Predictions = vec![];

                for j in 0..n {}
            }
        }

        self.fitted = true;

        Ok(())
    }

    pub fn predict(&self, x: &Matrix) -> Result<Predictions> {
        if !self.fitted {
            return Err(TesseractError::NotFitted);
        }

        let n = x.nrows();
        let d = x.ncols();

        if self.feature_index >= d {
            return Err(TesseractError::ShapeMismatch {
                expected: format!("feature_index < {}", d),
                got: format!("feature_index = {}", self.feature_index),
            });
        }

        let col = x.column(self.feature_index);
        let mut predictions = vec![0usize; n];

        let t = self.split_value;
        let left = self.left_class;
        let right = self.right_class;

        for (i, out) in predictions.iter_mut().enumerate() {
            *out = if col[i] <= t { left } else { right };
        }

        Ok(predictions)
    }
}
