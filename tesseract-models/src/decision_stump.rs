use tesseract_core::{Float, Label, Matrix, Predictions, Result, TesseractError, gini_from_counts};

/// A **decision stump**: a decision tree with depth 1 (a single split).
///
/// A stump learns:
/// - one feature index `j`
/// - one threshold `t`
///
/// and predicts a class label by routing each sample:
///
/// ```text
/// if x[j] <= t  => predict left_class
/// else         => predict right_class
/// ```
///
/// This implementation trains using a **CART-style** split search:
/// - For each feature, it sorts samples by feature value and performs a linear sweep.
/// - It selects the split that **minimizes weighted Gini impurity** of the two children.
///
/// # Notes
///
/// - This stump is for **classification** (labels are discrete).
/// - Training assumes labels are **dense** in `0..C-1` (so counts can be stored in a `Vec`).
/// - If the input contains `NaN` values, training returns [`TesseractError::InvalidValue`].
///
/// # Fields
///
/// - `feature_index`: The selected feature to split on.
/// - `feature_threshold`: The selected threshold value.
/// - `left_class`: The predicted label for samples with `x[feature_index] <= feature_threshold`.
/// - `right_class`: The predicted label for samples with `x[feature_index] > feature_threshold`.
/// - `fitted`: Whether [`fit`](DecisionStump::fit) has been called successfully.
#[derive(Debug, Clone)]
pub struct DecisionStump {
    /// Index of the feature used for splitting.
    feature_index: usize,
    /// Threshold used for splitting on `feature_index`.
    feature_threshold: Float,
    /// Majority class predicted for the left branch (`<= threshold`).
    left_class: usize,
    /// Majority class predicted for the right branch (`> threshold`).
    right_class: usize,
    /// Whether the model has been trained.
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
    /// Creates a new, unfitted [`DecisionStump`].
    ///
    /// This is equivalent to [`DecisionStump::default`]. The returned stump must be trained
    /// with [`fit`](DecisionStump::fit) before calling [`predict`](DecisionStump::predict).
    pub fn new() -> Self {
        Self::default()
    }

    /// Fits the decision stump on training data `x` and labels `y`.
    ///
    /// # Algorithm
    ///
    /// For each feature (column) `j`:
    /// 1. Build a list of `(feature_value, label)` pairs for all samples.
    /// 2. Sort pairs by `feature_value`.
    /// 3. Sweep candidate split points between adjacent distinct feature values.
    /// 4. For each candidate split, compute the **weighted Gini impurity**:
    ///
    /// ```text
    /// score = (n_left / n) * Gini(left) + (n_right / n) * Gini(right)
    /// ```
    ///
    /// The model chooses the split with the **minimum** score.
    ///
    /// The predicted leaf labels (`left_class`, `right_class`) are chosen as the
    /// **majority label** on each side of the best split.
    ///
    /// # Parameters
    ///
    /// - `x`: Feature matrix of shape `(n_samples, n_features)`.
    /// - `y`: Labels of length `n_samples`. Labels are assumed to be dense in `0..C-1`.
    ///
    /// # Errors
    ///
    /// - [`TesseractError::EmptyTrainingData`] if `x` has zero rows.
    /// - [`TesseractError::ShapeMismatch`] if `y.len() != x.nrows()`.
    /// - [`TesseractError::InvalidTrainingData`] if labels are inconsistent or empty.
    /// - [`TesseractError::InvalidValue`] if any feature value is `NaN`.
    ///
    /// # Complexity
    ///
    /// Let `n = n_samples`, `d = n_features`.
    /// - Time: `O(d * n log n)` due to sorting per feature.
    /// - Space: `O(n + C)` for the pair buffer and class counts.
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

        // Reused buffers (avoid repeated allocation per feature)
        let mut pairs: Vec<(Float, Label)> = Vec::with_capacity(n);
        let mut left_counts = vec![0usize; num_classes];
        let mut right_counts = vec![0usize; num_classes];

        for feature in 0..d {
            pairs.clear();

            // Build (value, label) for this feature and validate values.
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

            // Sort by feature value (CART-style sweep)
            pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

            // Init counts: all samples start on the right
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

                // Move one sample from right -> left
                left_counts[label_i] += 1;
                right_counts[label_i] -= 1;
                n_left += 1;
                n_right -= 1;

                // Only consider thresholds between distinct values
                let v_next = pairs[i + 1].0;
                if v_i == v_next {
                    continue;
                }

                // Weighted Gini score for this candidate split
                let g_l = gini_from_counts(&left_counts, n_left);
                let g_r = gini_from_counts(&right_counts, n_right);

                let w_l = n_left as Float / n as Float;
                let w_r = n_right as Float / n as Float;
                let score = w_l * g_l + w_r * g_r;

                if score < best_score {
                    best_score = score;
                    best_feature = feature;
                    best_threshold = (v_i + v_next) * 0.5;

                    // Leaf predictions: majority class on each side
                    best_left_label = argmax_count(&left_counts);
                    best_right_label = argmax_count(&right_counts);
                }
            }
        }

        self.feature_index = best_feature;
        self.feature_threshold = best_threshold;
        self.left_class = best_left_label;
        self.right_class = best_right_label;
        self.fitted = true;

        Ok(())
    }

    /// Predicts class labels for input matrix `x`.
    ///
    /// Each sample is routed by the learned split:
    ///
    /// ```text
    /// if x[feature_index] <= feature_threshold  => left_class
    /// else                                      => right_class
    /// ```
    ///
    /// # Parameters
    ///
    /// - `x`: Feature matrix of shape `(n_samples, n_features)`.
    ///
    /// # Returns
    ///
    /// A vector of predicted labels of length `n_samples`.
    ///
    /// # Errors
    ///
    /// - [`TesseractError::NotFitted`] if the stump has not been trained.
    /// - [`TesseractError::ShapeMismatch`] if `feature_index` is out of bounds for `x`.
    ///
    /// # Complexity
    ///
    /// - Time: `O(n_samples)` (one comparison per sample).
    /// - Space: `O(n_samples)` for the output vector.
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

/// Returns the index of the maximum count (the majority class).
///
/// This is used to choose the predicted leaf label after a split.
///
/// # Parameters
///
/// - `counts`: A slice where `counts[k]` is the number of samples in class `k`.
///
/// # Returns
///
/// The class index `k` with the largest count. If multiple classes tie for the
/// maximum count, this returns the **first** one encountered (smallest label).
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
