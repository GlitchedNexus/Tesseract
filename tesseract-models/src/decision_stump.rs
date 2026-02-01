#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use tesseract_core::{Float, Label, Matrix, Predictions, Result, TesseractError, gini_from_counts};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

        // Structure to hold the best split for a single feature
        #[derive(Debug, Clone)]
        #[allow(dead_code)]
        struct FeatureSplit {
            feature: usize,
            threshold: Float,
            score: Float,
            left_label: Label,
            right_label: Label,
        }

        #[cfg(feature = "parallel")]
        let best_split = (0..d)
            .into_par_iter()
            .map(|feature| {
                let mut pairs: Vec<(Float, Label)> = Vec::with_capacity(n);
                let mut left_counts = vec![0usize; num_classes];
                let mut right_counts = vec![0usize; num_classes];

                // Build (value, label) for this feature
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
                let mut feature_best_score = Float::INFINITY;
                let mut feature_best_threshold = 0.0;
                let mut feature_best_left_label = 0;
                let mut feature_best_right_label = 0;

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

                    if score < feature_best_score {
                        feature_best_score = score;
                        feature_best_threshold = (v_i + v_next) * 0.5;
                        feature_best_left_label = argmax_count(&left_counts);
                        feature_best_right_label = argmax_count(&right_counts);
                    }
                }

                Ok(FeatureSplit {
                    feature,
                    threshold: feature_best_threshold,
                    score: feature_best_score,
                    left_label: feature_best_left_label,
                    right_label: feature_best_right_label,
                })
            })
            .collect::<Result<Vec<FeatureSplit>>>()?
            .into_iter()
            .min_by(|a, b| a.score.total_cmp(&b.score))
            .unwrap();

        #[cfg(not(feature = "parallel"))]
        let best_split = {
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

            FeatureSplit {
                feature: best_feature,
                threshold: best_threshold,
                score: best_score,
                left_label: best_left_label,
                right_label: best_right_label,
            }
        };

        self.feature_index = best_split.feature;
        self.feature_threshold = best_split.threshold;
        self.left_class = best_split.left_label;
        self.right_class = best_split.right_label;
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
        let t = self.feature_threshold;
        let left = self.left_class;
        let right = self.right_class;

        #[cfg(feature = "parallel")]
        let predictions: Predictions = (0..n)
            .into_par_iter()
            .map(|i| if col[i] <= t { left } else { right })
            .collect();

        #[cfg(not(feature = "parallel"))]
        let predictions: Predictions = {
            let mut preds = vec![0usize; n];
            for (i, out) in preds.iter_mut().enumerate() {
                *out = if col[i] <= t { left } else { right };
            }
            preds
        };

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

#[cfg(test)]
mod tests {
    use super::*;
    use tesseract_core::Matrix;

    /// Helper to create a simple 2D matrix from nested arrays.
    fn matrix_from_vec(data: Vec<Vec<f32>>) -> Matrix {
        let nrows = data.len();
        let ncols = if nrows > 0 { data[0].len() } else { 0 };
        Matrix::from_fn(nrows, ncols, |i, j| data[i][j])
    }

    #[test]
    fn test_decision_stump_not_fitted() {
        let stump = DecisionStump::new();
        let x = matrix_from_vec(vec![vec![1.0, 2.0]]);
        let result = stump.predict(&x);
        assert!(matches!(result, Err(TesseractError::NotFitted)));
    }

    #[test]
    fn test_decision_stump_empty_training_data() {
        let mut stump = DecisionStump::new();
        let x_train = matrix_from_vec(vec![]);
        let y_train = vec![];
        let result = stump.fit(&x_train, &y_train);
        assert!(matches!(result, Err(TesseractError::EmptyTrainingData)));
    }

    #[test]
    fn test_decision_stump_label_mismatch() {
        let mut stump = DecisionStump::new();
        let x_train = matrix_from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let y_train = vec![0]; // wrong length
        let result = stump.fit(&x_train, &y_train);
        assert!(matches!(result, Err(TesseractError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_decision_stump_with_nan() {
        let mut stump = DecisionStump::new();
        let x_train = matrix_from_vec(vec![vec![1.0, 2.0], vec![f32::NAN, 4.0]]);
        let y_train = vec![0, 1];
        let result = stump.fit(&x_train, &y_train);
        assert!(matches!(result, Err(TesseractError::InvalidValue { .. })));
    }

    #[test]
    fn test_decision_stump_simple_split() {
        let mut stump = DecisionStump::new();
        // Clear separation: x[0] <= 5 => class 0, x[0] > 5 => class 1
        let x_train = matrix_from_vec(vec![
            vec![1.0, 0.0],
            vec![2.0, 0.0],
            vec![3.0, 0.0],
            vec![8.0, 0.0],
            vec![9.0, 0.0],
            vec![10.0, 0.0],
        ]);
        let y_train = vec![0, 0, 0, 1, 1, 1];
        stump.fit(&x_train, &y_train).unwrap();

        // Test predictions
        let x_test = matrix_from_vec(vec![vec![1.5, 0.0], vec![9.5, 0.0]]);
        let result = stump.predict(&x_test).unwrap();
        assert_eq!(result[0], 0); // should be left class
        assert_eq!(result[1], 1); // should be right class
    }

    #[test]
    fn test_decision_stump_feature_selection() {
        let mut stump = DecisionStump::new();
        // Feature 1 is informative, feature 0 is not
        let x_train = matrix_from_vec(vec![
            vec![0.0, 1.0],
            vec![0.0, 2.0],
            vec![0.0, 3.0],
            vec![0.0, 8.0],
            vec![0.0, 9.0],
            vec![0.0, 10.0],
        ]);
        let y_train = vec![0, 0, 0, 1, 1, 1];
        stump.fit(&x_train, &y_train).unwrap();

        // Should select feature 1
        assert_eq!(stump.feature_index, 1);

        let x_test = matrix_from_vec(vec![
            vec![100.0, 2.0], // feature 0 is ignored
            vec![-50.0, 9.0],
        ]);
        let result = stump.predict(&x_test).unwrap();
        assert_eq!(result[0], 0);
        assert_eq!(result[1], 1);
    }

    #[test]
    fn test_decision_stump_binary_classification() {
        let mut stump = DecisionStump::new();
        // XOR-like but linearly separable on one axis
        let x_train = matrix_from_vec(vec![
            vec![1.0, 1.0],
            vec![1.0, 2.0],
            vec![9.0, 1.0],
            vec![9.0, 2.0],
        ]);
        let y_train = vec![0, 0, 1, 1];
        stump.fit(&x_train, &y_train).unwrap();

        let x_test = matrix_from_vec(vec![vec![1.0, 1.5], vec![9.0, 1.5]]);
        let result = stump.predict(&x_test).unwrap();
        assert_eq!(result[0], 0);
        assert_eq!(result[1], 1);
    }

    #[test]
    fn test_decision_stump_multiclass() {
        let mut stump = DecisionStump::new();
        // Three classes
        let x_train = matrix_from_vec(vec![
            vec![1.0],
            vec![2.0],
            vec![5.0],
            vec![6.0],
            vec![10.0],
            vec![11.0],
        ]);
        let y_train = vec![0, 0, 1, 1, 2, 2];
        stump.fit(&x_train, &y_train).unwrap();

        // Stump can only split into 2 groups
        // It should find the best split (likely around 3.5 or 8)
        let x_test = matrix_from_vec(vec![vec![1.5], vec![5.5], vec![10.5]]);
        let result = stump.predict(&x_test).unwrap();
        // Exact predictions depend on which split was chosen
        // Just verify we get valid class labels
        assert!(result.iter().all(|&label| label <= 2));
    }

    #[test]
    fn test_decision_stump_single_class() {
        let mut stump = DecisionStump::new();
        // All same class - no useful split possible
        let x_train = matrix_from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
        let y_train = vec![0, 0, 0];
        stump.fit(&x_train, &y_train).unwrap();

        let x_test = matrix_from_vec(vec![vec![2.0, 3.0]]);
        let result = stump.predict(&x_test).unwrap();
        // Should predict the only class
        assert_eq!(result[0], 0);
    }

    #[test]
    fn test_decision_stump_threshold_placement() {
        let mut stump = DecisionStump::new();
        // Test that threshold is placed between values
        let x_train = matrix_from_vec(vec![vec![1.0], vec![2.0], vec![8.0], vec![9.0]]);
        let y_train = vec![0, 0, 1, 1];
        stump.fit(&x_train, &y_train).unwrap();

        // Threshold should be between 2.0 and 8.0
        assert!(stump.feature_threshold > 2.0);
        assert!(stump.feature_threshold < 8.0);
    }

    #[test]
    fn test_decision_stump_perfect_split() {
        let mut stump = DecisionStump::new();
        // Perfect Gini split
        let x_train = matrix_from_vec(vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![7.0],
            vec![8.0],
            vec![9.0],
        ]);
        let y_train = vec![0, 0, 0, 1, 1, 1];
        stump.fit(&x_train, &y_train).unwrap();

        // Should achieve perfect separation
        let x_test = matrix_from_vec(vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![7.0],
            vec![8.0],
            vec![9.0],
        ]);
        let result = stump.predict(&x_test).unwrap();
        assert_eq!(result, vec![0, 0, 0, 1, 1, 1]);
    }

    #[test]
    fn test_decision_stump_duplicate_values() {
        let mut stump = DecisionStump::new();
        // Multiple samples with same feature value but different labels
        let x_train = matrix_from_vec(vec![
            vec![1.0],
            vec![1.0],
            vec![1.0],
            vec![5.0],
            vec![5.0],
            vec![5.0],
        ]);
        let y_train = vec![0, 0, 0, 1, 1, 1];
        stump.fit(&x_train, &y_train).unwrap();

        let x_test = matrix_from_vec(vec![vec![1.0], vec![5.0]]);
        let result = stump.predict(&x_test).unwrap();
        assert_eq!(result[0], 0);
        assert_eq!(result[1], 1);
    }

    #[test]
    fn test_decision_stump_noisy_data() {
        let mut stump = DecisionStump::new();
        // Mostly separable with some noise
        let x_train = matrix_from_vec(vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0], // noisy sample
            vec![7.0], // noisy sample
            vec![8.0],
            vec![9.0],
            vec![10.0],
        ]);
        let y_train = vec![0, 0, 0, 1, 0, 1, 1, 1];
        stump.fit(&x_train, &y_train).unwrap();

        // Should still find a reasonable split
        let x_test = matrix_from_vec(vec![vec![2.0], vec![9.0]]);
        let result = stump.predict(&x_test).unwrap();
        // Predictions should be reasonable given the data
        assert!(result[0] == 0 || result[0] == 1);
        assert!(result[1] == 0 || result[1] == 1);
    }

    #[test]
    fn test_decision_stump_predict_shape_mismatch() {
        let mut stump = DecisionStump::new();
        let x_train = matrix_from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let y_train = vec![0, 1];
        stump.fit(&x_train, &y_train).unwrap();

        // Test data with wrong number of features
        let x_test = matrix_from_vec(vec![vec![1.0]]);
        // feature_index might be >= ncols for test data
        let result = stump.predict(&x_test);
        // This might fail with ShapeMismatch or succeed depending on which feature was selected
        if stump.feature_index >= 1 {
            assert!(matches!(result, Err(TesseractError::ShapeMismatch { .. })));
        }
    }

    #[test]
    fn test_decision_stump_multiple_predictions() {
        let mut stump = DecisionStump::new();
        let x_train = matrix_from_vec(vec![vec![1.0], vec![2.0], vec![8.0], vec![9.0]]);
        let y_train = vec![0, 0, 1, 1];
        stump.fit(&x_train, &y_train).unwrap();

        // Multiple test samples
        let x_test = matrix_from_vec(vec![vec![1.5], vec![8.5], vec![5.0], vec![0.5]]);
        let result = stump.predict(&x_test).unwrap();
        assert_eq!(result.len(), 4);
        // Verify each prediction is valid
        assert!(result.iter().all(|&label| label <= 1));
    }

    #[test]
    fn test_decision_stump_1d_edge_case() {
        let mut stump = DecisionStump::new();
        // Minimal case: 2 samples, 2 classes
        let x_train = matrix_from_vec(vec![vec![1.0], vec![2.0]]);
        let y_train = vec![0, 1];
        stump.fit(&x_train, &y_train).unwrap();

        let x_test = matrix_from_vec(vec![vec![1.0], vec![2.0]]);
        let result = stump.predict(&x_test).unwrap();
        assert_eq!(result[0], 0);
        assert_eq!(result[1], 1);
    }

    #[test]
    fn test_decision_stump_high_dimensional() {
        let mut stump = DecisionStump::new();
        // 5 features, but only feature 2 is informative
        let x_train = matrix_from_vec(vec![
            vec![0.0, 0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 2.0, 0.0, 0.0],
            vec![0.0, 0.0, 3.0, 0.0, 0.0],
            vec![0.0, 0.0, 8.0, 0.0, 0.0],
            vec![0.0, 0.0, 9.0, 0.0, 0.0],
            vec![0.0, 0.0, 10.0, 0.0, 0.0],
        ]);
        let y_train = vec![0, 0, 0, 1, 1, 1];
        stump.fit(&x_train, &y_train).unwrap();

        // Should select feature 2
        assert_eq!(stump.feature_index, 2);

        let x_test = matrix_from_vec(vec![
            vec![100.0, 100.0, 2.5, 100.0, 100.0],
            vec![-50.0, -50.0, 8.5, -50.0, -50.0],
        ]);
        let result = stump.predict(&x_test).unwrap();
        assert_eq!(result[0], 0);
        assert_eq!(result[1], 1);
    }

    #[test]
    fn test_decision_stump_imbalanced_classes() {
        let mut stump = DecisionStump::new();
        // Imbalanced: 5 samples of class 0, 1 sample of class 1
        let x_train = matrix_from_vec(vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
            vec![10.0],
        ]);
        let y_train = vec![0, 0, 0, 0, 0, 1];
        stump.fit(&x_train, &y_train).unwrap();

        let x_test = matrix_from_vec(vec![vec![3.0], vec![10.0]]);
        let result = stump.predict(&x_test).unwrap();
        // Should still make reasonable predictions
        assert!(result.iter().all(|&label| label <= 1));
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_decision_stump_serialize_deserialize_json() {
        let mut stump = DecisionStump::new();
        let x_train = matrix_from_vec(vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![8.0, 9.0],
            vec![9.0, 10.0],
        ]);
        let y_train = vec![0, 0, 1, 1];
        stump.fit(&x_train, &y_train).unwrap();

        // Serialize to JSON
        let serialized = serde_json::to_string(&stump).expect("Failed to serialize");
        assert!(!serialized.is_empty());

        // Deserialize from JSON
        let deserialized: DecisionStump = serde_json::from_str(&serialized).expect("Failed to deserialize");

        // Test that deserialized model works
        let x_test = matrix_from_vec(vec![vec![1.5, 2.5], vec![8.5, 9.5]]);
        let result_original = stump.predict(&x_test).unwrap();
        let result_deserialized = deserialized.predict(&x_test).unwrap();

        assert_eq!(result_original, result_deserialized);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_decision_stump_serialize_deserialize_binary() {
        let mut stump = DecisionStump::new();
        let x_train = matrix_from_vec(vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
            vec![7.0],
            vec![8.0],
            vec![9.0],
        ]);
        let y_train = vec![0, 0, 0, 1, 1, 1];
        stump.fit(&x_train, &y_train).unwrap();

        // Serialize using serde (binary format via JSON bytes)
        let serialized = serde_json::to_vec(&stump).expect("Failed to serialize");
        assert!(!serialized.is_empty());

        // Deserialize
        let deserialized: DecisionStump = serde_json::from_slice(&serialized).expect("Failed to deserialize");

        // Verify predictions match
        let x_test = matrix_from_vec(vec![vec![1.5], vec![8.5]]);
        let result_original = stump.predict(&x_test).unwrap();
        let result_deserialized = deserialized.predict(&x_test).unwrap();

        assert_eq!(result_original, result_deserialized);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_decision_stump_serialize_unfitted() {
        let stump = DecisionStump::new();

        // Should be able to serialize unfitted model
        let serialized = serde_json::to_string(&stump).expect("Failed to serialize");
        let deserialized: DecisionStump = serde_json::from_str(&serialized).expect("Failed to deserialize");

        // Both should fail with NotFitted
        let x_test = matrix_from_vec(vec![vec![1.0, 2.0]]);
        assert!(matches!(stump.predict(&x_test), Err(TesseractError::NotFitted)));
        assert!(matches!(deserialized.predict(&x_test), Err(TesseractError::NotFitted)));
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_decision_stump_roundtrip_preserves_state() {
        let mut stump = DecisionStump::new();
        let x_train = matrix_from_vec(vec![
            vec![1.0, 5.0],
            vec![2.0, 6.0],
            vec![8.0, 1.0],
            vec![9.0, 2.0],
        ]);
        let y_train = vec![0, 0, 1, 1];
        stump.fit(&x_train, &y_train).unwrap();

        // Serialize and deserialize
        let serialized = serde_json::to_string(&stump).unwrap();
        let deserialized: DecisionStump = serde_json::from_str(&serialized).unwrap();

        // All internal state should be preserved
        assert_eq!(stump.feature_index, deserialized.feature_index);
        assert_eq!(stump.feature_threshold, deserialized.feature_threshold);
        assert_eq!(stump.left_class, deserialized.left_class);
        assert_eq!(stump.right_class, deserialized.right_class);
        assert_eq!(stump.fitted, deserialized.fitted);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_decision_stump_clone_and_serialize() {
        let mut stump = DecisionStump::new();
        let x_train = matrix_from_vec(vec![
            vec![1.0],
            vec![2.0],
            vec![8.0],
            vec![9.0],
        ]);
        let y_train = vec![0, 0, 1, 1];
        stump.fit(&x_train, &y_train).unwrap();

        // Clone the stump
        let cloned = stump.clone();

        // Serialize both
        let serialized_original = serde_json::to_string(&stump).unwrap();
        let serialized_cloned = serde_json::to_string(&cloned).unwrap();

        // Should produce identical serialization
        assert_eq!(serialized_original, serialized_cloned);

        // Both should make identical predictions
        let x_test = matrix_from_vec(vec![vec![1.5], vec![8.5]]);
        let result_original = stump.predict(&x_test).unwrap();
        let result_cloned = cloned.predict(&x_test).unwrap();

        assert_eq!(result_original, result_cloned);
    }
}
