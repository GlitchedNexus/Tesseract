use tesseract_core::{Float, Label, Matrix, Predictions, Result, TesseractError, gini_from_counts};

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
        if n == 0 {
            return Err(TesseractError::EmptyTrainingData);
        }
        let d = x.ncols();

        if y.len() != n {
            return Err(TesseractError::ShapeMismatch {
                expected: format!("Expected {} sample labels", n),
                got: format!("Got {} sample labels", y.len()),
            });
        }

        // --- Determine number of classes (assumes dense 0..C-1) ---
        let &max_label = y.iter().max().ok_or(TesseractError::InvalidTrainingData)?;
        let num_classes = max_label + 1;

        let mut best_feature = 0usize;
        let mut best_threshold: Float = 0.0;
        let mut best_score: Float = Float::INFINITY;
        let mut best_left_label: Label = 0;
        let mut best_right_label: Label = 0;

        // Reused buffers
        let mut pairs: Vec<(Float, Label)> = Vec::with_capacity(n);
        let mut left_counts = vec![0usize; num_classes];
        let mut right_counts = vec![0usize; num_classes];

        for feature in 0..d {
            pairs.clear();

            // Build (value, label) for this feature
            // Also validate NaNs once here.
            let col = x.column(feature);
            for i in 0..n {
                let v = col[i];
                if v.is_nan() {
                    return Err(TesseractError::InvalidValue {
                        message: "Expected float got NaN.".into(),
                    });
                }
                pairs.push((v, y[i]));
            }

            // Sort by feature value
            pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

            // Init counts: all samples on the right initially
            left_counts.fill(0);
            right_counts.fill(0);
            for &(_, label) in &pairs {
                if label >= num_classes {
                    return Err(TesseractError::InvalidTrainingData);
                }
                right_counts[label] += 1;
            }

            let mut n_left = 0usize;
            let mut n_right = n;

            // Sweep split point between i and i+1
            for i in 0..(n - 1) {
                let (v_i, label_i) = pairs[i];

                // Move sample i from right -> left
                left_counts[label_i] += 1;
                right_counts[label_i] -= 1;
                n_left += 1;
                n_right -= 1;

                let v_next = pairs[i + 1].0;
                if v_i == v_next {
                    continue; // no threshold between equal values
                }

                // Compute weighted gini
                let g_l = gini_from_counts(&left_counts, n_left);
                let g_r = gini_from_counts(&right_counts, n_right);

                let w_l = n_left as Float / n as Float;
                let w_r = n_right as Float / n as Float;
                let score = w_l * g_l + w_r * g_r;

                if score < best_score {
                    best_score = score;
                    best_feature = feature;
                    best_threshold = (v_i + v_next) * 0.5;

                    // Majority labels for leaf predictions
                    best_left_label = argmax_count(&left_counts);
                    best_right_label = argmax_count(&right_counts);
                }
            }
        }

        self.feature_index = best_feature;
        self.feature_threshold = best_threshold; // or feature_threshold; keep one field
        self.left_class = best_left_label;
        self.right_class = best_right_label;
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

        let t = self.feature_threshold;
        let left = self.left_class;
        let right = self.right_class;

        for (i, out) in predictions.iter_mut().enumerate() {
            *out = if col[i] <= t { left } else { right };
        }

        Ok(predictions)
    }
}

fn argmax_count(counts: &[usize]) -> Label {
    let mut best_label = 0usize;
    let mut best = 0usize;
    for (label, &c) in counts.iter().enumerate() {
        if c > best {
            best = c;
            best_label = label;
        }
    }
    best_label
}
