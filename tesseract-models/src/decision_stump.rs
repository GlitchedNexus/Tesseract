use std::{collections::HashMap, fmt::Debug};
use tesseract_core::{Float, Label, Matrix, Predictions, Result, TesseractError};

#[derive(Debug, Clone)]
pub struct DecisionStump {
    feature_index: usize,
    feature_threshold: Float,
    left_class: usize,
    right_class: usize,
    fitted: bool,
}

impl Default for DecisionStump {
    fn default() -> Self {
        Self {
            feature_index: 1,
            feature_threshold: 0.0,
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

        let mut best_feature: usize = 0;
        let mut best_threshold: Float = 0.0;
        let mut best_gini: Float = Float::INFINITY;
        let mut best_labels: (Label, Label) = (0, 0);

        for feature in 0..d {
            let col = x.column(feature);

            for threshold in col {
                let mut left: Predictions = Vec::new();
                let mut right: Predictions = Vec::new();

                for i in 0..n {
                    match col.get(i) {
                        Some(val) => {
                            if val.is_nan() {
                                return Err(TesseractError::InvalidValue {
                                    message: String::from("Expected float got NaN."),
                                });
                            }

                            if val <= threshold {
                                left.push(i);
                            } else {
                                right.push(i);
                            }
                        }

                        None => {
                            return Err(TesseractError::InternalError);
                        }
                    }
                }

                let mut gini_left: f32 = 0.0;
                let n_left = left.len();

                let mut gini_right: f32 = 0.0;
                let n_right = right.len();

                let mut labels_left: HashMap<Label, usize> = HashMap::new();
                let mut labels_right: HashMap<Label, usize> = HashMap::new();

                for (_, index) in left.iter().enumerate() {
                    let label = y.get(*index);
                    match label {
                        Some(l) => {
                            let c = labels_left.entry(*l).or_insert(0);
                            *c += 1;
                        }
                        None => return Err(TesseractError::InvalidTrainingData),
                    }
                }

                for (_, index) in right.iter().enumerate() {
                    let label = y.get(*index);
                    match label {
                        Some(l) => {
                            let c = labels_right.entry(*l).or_insert(0);
                            *c += 1;
                        }
                        None => return Err(TesseractError::InvalidTrainingData),
                    }
                }

                let mut sum_sq = 0.0;
                for &count in labels_left.values() {
                    let p = count as Float / n_left as Float;
                    sum_sq += p * p;
                }
                let gini_left = 1.0 - sum_sq;

                let mut sum_sq = 0.0;
                for &count in labels_right.values() {
                    let p = count as Float / n_right as Float;
                    sum_sq += p * p;
                }
                let gini_right = 1.0 - sum_sq;

                let w_left = n_left as Float / n as Float;
                let w_right = n_right as Float / n as Float;
                let gini = w_left * gini_left + w_right * gini_right;

                if gini <= best_gini {
                    best_feature = feature;
                    best_threshold = *threshold;
                    best_gini = gini;

                    let (left_label, _) =
                        labels_left.iter().max_by_key(|&(_, count)| count).unwrap();

                    let (right_label, _) =
                        labels_right.iter().max_by_key(|&(_, count)| count).unwrap();

                    best_labels = (*left_label, *right_label)
                }
            }
        }

        self.feature_index = best_feature;
        self.feature_threshold = best_threshold;
        self.fitted = true;
        self.left_class = best_labels.0;
        self.right_class = best_labels.1;

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

        let t = self.feature_threshold;
        let left = self.left_class;
        let right = self.right_class;

        for (i, out) in predictions.iter_mut().enumerate() {
            *out = if col[i] <= t { left } else { right };
        }

        Ok(predictions)
    }
}
