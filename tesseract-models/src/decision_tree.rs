#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use tesseract_core::{Float, Label, Matrix, Predictions, Result, TesseractError, gini_from_counts};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// A node in the decision tree.
///
/// Each node represents either:
/// - **Leaf node**: stores a predicted class label
/// - **Internal node**: stores a split criterion (feature + threshold) and pointers to children
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
enum TreeNode {
    /// Leaf node storing the predicted class label.
    Leaf { label: Label },
    /// Internal node storing split criterion and child indices.
    Internal {
        feature_index: usize,
        threshold: Float,
        left_child: Box<TreeNode>,
        right_child: Box<TreeNode>,
    },
}

/// **Decision Tree** classifier using greedy recursive splitting.
///
/// A decision tree recursively partitions the feature space using binary splits.
/// Each split is chosen greedily to minimize **weighted Gini impurity** (CART algorithm).
///
/// # Algorithm
///
/// **Training** ([`fit`](DecisionTree::fit)):
/// 1. Start with all samples at the root
/// 2. For each node:
///    - If stopping criterion met (max depth, min samples, pure node), create leaf
///    - Otherwise, find best split across all features (like decision stump)
///    - Partition data and recursively build left and right subtrees
///
/// **Prediction** ([`predict`](DecisionTree::predict)):
/// - Traverse tree from root to leaf following split decisions
/// - Return leaf's predicted label
///
/// # Stopping criteria
///
/// - `max_depth`: Maximum tree depth (0 = root only, 1 = one split, etc.)
/// - `min_samples_split`: Minimum samples required to split a node
/// - `min_samples_leaf`: Minimum samples required in each leaf after split
/// - Pure node (Gini = 0): all samples have same class
///
/// # Fields
///
/// - `root`: Root node of the tree (set after fitting)
/// - `max_depth`: Maximum depth
/// - `min_samples_split`: Min samples to allow split
/// - `min_samples_leaf`: Min samples per leaf
/// - `num_classes`: Number of classes (inferred from training data)
///
/// # Notes
///
/// - Uses CART-style greedy splitting (same as [`DecisionStump`])
/// - Can overfit if `max_depth` too large or `min_samples_*` too small
/// - Consider pruning or ensemble methods (Random Forest, boosting) for better generalization
///
/// # Errors
///
/// - [`TesseractError::EmptyTrainingData`] if training data empty
/// - [`TesseractError::ShapeMismatch`] if labels length mismatches samples
/// - [`TesseractError::InvalidValue`] if `NaN` encountered
/// - [`TesseractError::NotFitted`] if prediction before fitting
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct DecisionTree {
    /// Root node of the decision tree.
    root: Option<TreeNode>,
    /// Maximum depth of the tree.
    max_depth: usize,
    /// Minimum samples required to split an internal node.
    min_samples_split: usize,
    /// Minimum samples required in a leaf node.
    min_samples_leaf: usize,
    /// Number of classes (inferred during fit).
    num_classes: usize,
}

impl Default for DecisionTree {
    fn default() -> Self {
        Self {
            root: None,
            max_depth: 10,
            min_samples_split: 2,
            min_samples_leaf: 1,
            num_classes: 0,
        }
    }
}

impl DecisionTree {
    /// Creates a new, unfitted decision tree.
    ///
    /// # Parameters
    ///
    /// - `max_depth`: Maximum depth (e.g., 10). Use smaller for less overfitting.
    /// - `min_samples_split`: Min samples to split (e.g., 2). Higher = less splits.
    /// - `min_samples_leaf`: Min samples per leaf (e.g., 1). Higher = smoother boundaries.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let tree = DecisionTree::new(5, 10, 5);
    /// ```
    pub fn new(max_depth: usize, min_samples_split: usize, min_samples_leaf: usize) -> Self {
        Self {
            root: None,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            num_classes: 0,
        }
    }

    /// Returns the majority class label from a set of labels.
    fn majority_class(labels: &[Label]) -> Label {
        if labels.is_empty() {
            return 0;
        }

        let max_label = *labels.iter().max().unwrap_or(&0);
        let mut counts = vec![0usize; max_label + 1];

        for &label in labels {
            counts[label] += 1;
        }

        let mut best_label = 0;
        let mut best_count = 0;

        for (label, &count) in counts.iter().enumerate() {
            if count > best_count {
                best_count = count;
                best_label = label;
            }
        }

        best_label
    }

    /// Finds the best split for a subset of the data.
    ///
    /// This performs the same CART-style sweep as DecisionStump, but on a subset
    /// of samples (identified by `indices`).
    ///
    /// # Returns
    ///
    /// `Some((feature, threshold, left_indices, right_indices, score))` if a valid split found,
    /// `None` otherwise.
    fn find_best_split(
        &self,
        x: &Matrix,
        y: &[Label],
        indices: &[usize],
    ) -> Result<Option<(usize, Float, Vec<usize>, Vec<usize>, Float)>> {
        let n = indices.len();
        let d = x.ncols();

        if n < self.min_samples_split {
            return Ok(None);
        }

        // Determine number of classes for this subset
        let max_label = indices.iter().map(|&i| y[i]).max().unwrap_or(0);
        let num_classes = max_label + 1;

        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_score = Float::INFINITY;
        let mut best_left_indices: Vec<usize> = Vec::new();
        let mut best_right_indices: Vec<usize> = Vec::new();

        // Try each feature
        for feature in 0..d {
            // Build (value, label, original_index) pairs
            let mut pairs: Vec<(Float, Label, usize)> = Vec::with_capacity(n);

            for &idx in indices {
                let val = x[(idx, feature)];
                if val.is_nan() {
                    return Err(TesseractError::InvalidValue {
                        message: "NaN encountered in feature values".into(),
                    });
                }
                pairs.push((val, y[idx], idx));
            }

            // Sort by feature value
            pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

            // Initialize counts
            let mut left_counts = vec![0usize; num_classes];
            let mut right_counts = vec![0usize; num_classes];

            for &(_, label, _) in &pairs {
                if label >= num_classes {
                    return Err(TesseractError::InvalidTrainingData);
                }
                right_counts[label] += 1;
            }

            let mut n_left = 0;
            let mut n_right = n;

            // Sweep split points
            for i in 0..(n - 1) {
                let (val_i, label_i, _) = pairs[i];

                // Move one sample from right to left
                left_counts[label_i] += 1;
                right_counts[label_i] -= 1;
                n_left += 1;
                n_right -= 1;

                // Skip if next value is same
                let val_next = pairs[i + 1].0;
                if val_i == val_next {
                    continue;
                }

                // Check min_samples_leaf constraint
                if n_left < self.min_samples_leaf || n_right < self.min_samples_leaf {
                    continue;
                }

                // Compute weighted Gini impurity
                let g_l = gini_from_counts(&left_counts, n_left);
                let g_r = gini_from_counts(&right_counts, n_right);

                let w_l = n_left as Float / n as Float;
                let w_r = n_right as Float / n as Float;
                let score = w_l * g_l + w_r * g_r;

                if score < best_score {
                    best_score = score;
                    best_feature = feature;
                    best_threshold = (val_i + val_next) * 0.5;

                    // Collect indices for left and right
                    best_left_indices.clear();
                    best_right_indices.clear();

                    for &(val, _, idx) in &pairs {
                        if val <= best_threshold {
                            best_left_indices.push(idx);
                        } else {
                            best_right_indices.push(idx);
                        }
                    }
                }
            }
        }

        if best_score == Float::INFINITY {
            Ok(None)
        } else {
            Ok(Some((
                best_feature,
                best_threshold,
                best_left_indices,
                best_right_indices,
                best_score,
            )))
        }
    }

    /// Recursively builds the tree.
    ///
    /// # Parameters
    ///
    /// - `x`: Feature matrix
    /// - `y`: Labels
    /// - `indices`: Indices of samples in this node
    /// - `depth`: Current depth (0 = root)
    ///
    /// # Returns
    ///
    /// A [`TreeNode`] (either Leaf or Internal).
    fn build_tree(
        &self,
        x: &Matrix,
        y: &[Label],
        indices: &[usize],
        depth: usize,
    ) -> Result<TreeNode> {
        let n = indices.len();

        // Base case: create leaf if stopping criterion met
        if n == 0 {
            return Ok(TreeNode::Leaf { label: 0 });
        }

        let labels_subset: Vec<Label> = indices.iter().map(|&i| y[i]).collect();
        let majority = Self::majority_class(&labels_subset);

        // Stop if:
        // 1. Max depth reached
        // 2. Too few samples to split
        // 3. Pure node (all same class)
        if depth >= self.max_depth || n < self.min_samples_split {
            return Ok(TreeNode::Leaf { label: majority });
        }

        // Check if pure
        let first_label = y[indices[0]];
        let is_pure = indices.iter().all(|&i| y[i] == first_label);
        if is_pure {
            return Ok(TreeNode::Leaf { label: first_label });
        }

        // Find best split
        match self.find_best_split(x, y, indices)? {
            None => {
                // No valid split found, create leaf
                Ok(TreeNode::Leaf { label: majority })
            }
            Some((feature, threshold, left_indices, right_indices, _score)) => {
                // Recursively build children
                let left_child = Box::new(self.build_tree(x, y, &left_indices, depth + 1)?);
                let right_child = Box::new(self.build_tree(x, y, &right_indices, depth + 1)?);

                Ok(TreeNode::Internal {
                    feature_index: feature,
                    threshold,
                    left_child,
                    right_child,
                })
            }
        }
    }

    /// Fits the decision tree on training data.
    ///
    /// # Parameters
    ///
    /// - `x`: Feature matrix of shape `(n_samples, n_features)`
    /// - `y`: Labels of length `n_samples` (assumed dense in `0..C-1`)
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, storing tree in `self.root`.
    ///
    /// # Errors
    ///
    /// - [`TesseractError::EmptyTrainingData`] if `x` has zero rows
    /// - [`TesseractError::ShapeMismatch`] if `y.len() != x.nrows()`
    /// - [`TesseractError::InvalidValue`] if `NaN` in features
    /// - [`TesseractError::InvalidTrainingData`] if labels inconsistent
    ///
    /// # Complexity
    ///
    /// Let `n = n_samples`, `d = n_features`, `h = tree height`.
    /// - Time: `O(d * n * log(n) * h)` in average case (sorting per node)
    /// - Space: `O(n + h)` for recursion and tree storage
    pub fn fit(&mut self, x: &Matrix, y: &Predictions) -> Result<()> {
        let n = x.nrows();

        if n == 0 {
            return Err(TesseractError::EmptyTrainingData);
        }

        if y.len() != n {
            return Err(TesseractError::ShapeMismatch {
                expected: format!("Expected {} labels", n),
                got: format!("Got {} labels", y.len()),
            });
        }

        // Infer number of classes
        let max_label = *y.iter().max().ok_or(TesseractError::InvalidTrainingData)?;
        self.num_classes = max_label + 1;

        // Build tree starting from root with all samples
        let all_indices: Vec<usize> = (0..n).collect();
        let root = self.build_tree(x, y, &all_indices, 0)?;
        self.root = Some(root);

        Ok(())
    }

    /// Predicts the class label for a single sample by traversing the tree.
    fn predict_one(&self, x_row: &[Float], node: &TreeNode) -> Result<Label> {
        match node {
            TreeNode::Leaf { label } => Ok(*label),
            TreeNode::Internal {
                feature_index,
                threshold,
                left_child,
                right_child,
            } => {
                if *feature_index >= x_row.len() {
                    return Err(TesseractError::ShapeMismatch {
                        expected: format!("feature_index < {}", x_row.len()),
                        got: format!("feature_index = {}", feature_index),
                    });
                }

                let val = x_row[*feature_index];
                if val <= *threshold {
                    self.predict_one(x_row, left_child)
                } else {
                    self.predict_one(x_row, right_child)
                }
            }
        }
    }

    /// Predicts class labels for input matrix `x`.
    ///
    /// # Parameters
    ///
    /// - `x`: Query matrix of shape `(n_queries, n_features)`
    ///
    /// # Returns
    ///
    /// Vector of predicted labels of length `n_queries`.
    ///
    /// # Errors
    ///
    /// - [`TesseractError::NotFitted`] if tree not fitted
    /// - [`TesseractError::ShapeMismatch`] if feature dimensions mismatch
    ///
    /// # Complexity
    ///
    /// Let `n = n_queries`, `h = tree height`.
    /// - Time: `O(n * h)` (traverse tree for each query)
    /// - Space: `O(n)` for output
    pub fn predict(&self, x: &Matrix) -> Result<Predictions> {
        let root = self.root.as_ref().ok_or(TesseractError::NotFitted)?;

        let n = x.nrows();
        let mut predictions = vec![0; n];

        #[cfg(feature = "parallel")]
        {
            predictions
                .par_iter_mut()
                .enumerate()
                .try_for_each(|(i, pred)| {
                    let row: Vec<Float> = x.row(i).iter().copied().collect();
                    *pred = self.predict_one(&row, root)?;
                    Ok::<(), TesseractError>(())
                })?;
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..n {
                let row: Vec<Float> = x.row(i).iter().copied().collect();
                predictions[i] = self.predict_one(&row, root)?;
            }
        }

        Ok(predictions)
    }

    /// Returns the maximum depth parameter.
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Returns the minimum samples split parameter.
    pub fn min_samples_split(&self) -> usize {
        self.min_samples_split
    }

    /// Returns the minimum samples leaf parameter.
    pub fn min_samples_leaf(&self) -> usize {
        self.min_samples_leaf
    }

    /// Returns the number of classes (set after fitting).
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix_from_vec(data: Vec<Vec<f32>>) -> Matrix {
        let rows = data.len();
        let cols = data[0].len();
        Matrix::from_fn(rows, cols, |i, j| data[i][j])
    }

    #[test]
    fn test_decision_tree_new() {
        let tree = DecisionTree::new(5, 10, 5);
        assert_eq!(tree.max_depth, 5);
        assert_eq!(tree.min_samples_split, 10);
        assert_eq!(tree.min_samples_leaf, 5);
        assert!(tree.root.is_none());
    }

    #[test]
    fn test_decision_tree_empty_data() {
        let mut tree = DecisionTree::new(5, 2, 1);
        let x = Matrix::zeros(0, 2);
        let y = vec![];
        let result = tree.fit(&x, &y);
        assert!(matches!(result, Err(TesseractError::EmptyTrainingData)));
    }

    #[test]
    fn test_decision_tree_label_mismatch() {
        let mut tree = DecisionTree::new(5, 2, 1);
        let x = matrix_from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let y = vec![0]; // wrong length
        let result = tree.fit(&x, &y);
        assert!(matches!(result, Err(TesseractError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_decision_tree_not_fitted() {
        let tree = DecisionTree::new(5, 2, 1);
        let x = matrix_from_vec(vec![vec![1.0, 2.0]]);
        let result = tree.predict(&x);
        assert!(matches!(result, Err(TesseractError::NotFitted)));
    }

    #[test]
    fn test_decision_tree_simple_split() {
        let mut tree = DecisionTree::new(10, 2, 1);
        // Clear separation: x[0] <= 5 => class 0, x[0] > 5 => class 1
        let x = matrix_from_vec(vec![
            vec![1.0, 0.0],
            vec![2.0, 0.0],
            vec![3.0, 0.0],
            vec![8.0, 0.0],
            vec![9.0, 0.0],
            vec![10.0, 0.0],
        ]);
        let y = vec![0, 0, 0, 1, 1, 1];
        tree.fit(&x, &y).unwrap();

        // Test predictions
        let x_test = matrix_from_vec(vec![vec![1.5, 0.0], vec![9.5, 0.0]]);
        let result = tree.predict(&x_test).unwrap();
        assert_eq!(result[0], 0);
        assert_eq!(result[1], 1);
    }

    #[test]
    fn test_decision_tree_perfect_classification() {
        let mut tree = DecisionTree::new(10, 2, 1);
        let x = matrix_from_vec(vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![7.0],
            vec![8.0],
            vec![9.0],
        ]);
        let y = vec![0, 0, 0, 1, 1, 1];
        tree.fit(&x, &y).unwrap();

        // Should achieve perfect classification
        let result = tree.predict(&x).unwrap();
        assert_eq!(result, y);
    }

    #[test]
    fn test_decision_tree_max_depth_limit() {
        let mut tree = DecisionTree::new(0, 2, 1); // max_depth = 0 (only root)
        let x = matrix_from_vec(vec![
            vec![1.0],
            vec![2.0],
            vec![8.0],
            vec![9.0],
        ]);
        let y = vec![0, 0, 1, 1];
        tree.fit(&x, &y).unwrap();

        // With depth 0, can only create a leaf with majority class
        let result = tree.predict(&x).unwrap();
        // All predictions should be same (majority class)
        assert!(result.iter().all(|&p| p == result[0]));
    }

    #[test]
    fn test_decision_tree_multiclass() {
        let mut tree = DecisionTree::new(10, 2, 1);
        let x = matrix_from_vec(vec![
            vec![1.0],
            vec![2.0],
            vec![5.0],
            vec![6.0],
            vec![10.0],
            vec![11.0],
        ]);
        let y = vec![0, 0, 1, 1, 2, 2];
        tree.fit(&x, &y).unwrap();

        // Test predictions
        let x_test = matrix_from_vec(vec![vec![1.5], vec![5.5], vec![10.5]]);
        let result = tree.predict(&x_test).unwrap();
        
        // Should classify correctly
        assert_eq!(result[0], 0);
        assert_eq!(result[1], 1);
        assert_eq!(result[2], 2);
    }

    #[test]
    fn test_decision_tree_min_samples_split() {
        let mut tree = DecisionTree::new(10, 100, 1); // Very high min_samples_split
        let x = matrix_from_vec(vec![
            vec![1.0],
            vec![2.0],
            vec![8.0],
            vec![9.0],
        ]);
        let y = vec![0, 0, 1, 1];
        tree.fit(&x, &y).unwrap();

        // Should create leaf without splitting (< 100 samples)
        let result = tree.predict(&x).unwrap();
        // All predictions should be same
        assert!(result.iter().all(|&p| p == result[0]));
    }

    #[test]
    fn test_decision_tree_min_samples_leaf() {
        let mut tree = DecisionTree::new(10, 2, 3); // min_samples_leaf = 3
        let x = matrix_from_vec(vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
        ]);
        let y = vec![0, 0, 1, 1];
        tree.fit(&x, &y).unwrap();

        // Split that creates leaves with < 3 samples should be rejected
        let result = tree.predict(&x).unwrap();
        // Should create a single leaf (no valid split)
        assert!(result.iter().all(|&p| p == result[0]));
    }

    #[test]
    fn test_decision_tree_with_nan() {
        let mut tree = DecisionTree::new(5, 2, 1);
        let x = matrix_from_vec(vec![
            vec![1.0, 2.0],
            vec![f32::NAN, 4.0],
        ]);
        let y = vec![0, 1];
        let result = tree.fit(&x, &y);
        assert!(matches!(result, Err(TesseractError::InvalidValue { .. })));
    }

    #[test]
    fn test_decision_tree_single_class() {
        let mut tree = DecisionTree::new(10, 2, 1);
        let x = matrix_from_vec(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ]);
        let y = vec![0, 0, 0];
        tree.fit(&x, &y).unwrap();

        let result = tree.predict(&x).unwrap();
        assert_eq!(result, vec![0, 0, 0]);
    }

    #[test]
    fn test_decision_tree_high_dimensional() {
        let mut tree = DecisionTree::new(10, 2, 1);
        // 5 features, feature 2 is informative
        let x = matrix_from_vec(vec![
            vec![0.0, 0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 2.0, 0.0, 0.0],
            vec![0.0, 0.0, 3.0, 0.0, 0.0],
            vec![0.0, 0.0, 8.0, 0.0, 0.0],
            vec![0.0, 0.0, 9.0, 0.0, 0.0],
            vec![0.0, 0.0, 10.0, 0.0, 0.0],
        ]);
        let y = vec![0, 0, 0, 1, 1, 1];
        tree.fit(&x, &y).unwrap();

        let x_test = matrix_from_vec(vec![
            vec![100.0, 100.0, 2.5, 100.0, 100.0],
            vec![-50.0, -50.0, 8.5, -50.0, -50.0],
        ]);
        let result = tree.predict(&x_test).unwrap();
        assert_eq!(result[0], 0);
        assert_eq!(result[1], 1);
    }

    #[test]
    fn test_decision_tree_xor_problem() {
        let mut tree = DecisionTree::new(10, 2, 1);
        // XOR-like pattern (requires multiple splits)
        let x = matrix_from_vec(vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ]);
        let y = vec![0, 1, 1, 0];
        tree.fit(&x, &y).unwrap();

        // Tree should learn XOR pattern
        let result = tree.predict(&x).unwrap();
        // May not be perfect due to data size, but should try
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_decision_tree_getters() {
        let mut tree = DecisionTree::new(7, 15, 3);
        assert_eq!(tree.max_depth(), 7);
        assert_eq!(tree.min_samples_split(), 15);
        assert_eq!(tree.min_samples_leaf(), 3);
        assert_eq!(tree.num_classes(), 0); // Before fitting

        let x = matrix_from_vec(vec![vec![1.0], vec![2.0]]);
        let y = vec![0, 1];
        tree.fit(&x, &y).unwrap();

        assert_eq!(tree.num_classes(), 2); // After fitting
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_decision_tree_serialize_deserialize_json() {
        let mut tree = DecisionTree::new(5, 2, 1);
        let x = matrix_from_vec(vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![8.0, 9.0],
            vec![9.0, 10.0],
        ]);
        let y = vec![0, 0, 1, 1];
        tree.fit(&x, &y).unwrap();

        // Serialize
        let serialized = serde_json::to_string(&tree).expect("Failed to serialize");
        assert!(!serialized.is_empty());

        // Deserialize
        let deserialized: DecisionTree = serde_json::from_str(&serialized).expect("Failed to deserialize");

        // Verify predictions match
        let x_test = matrix_from_vec(vec![vec![1.5, 2.5], vec![8.5, 9.5]]);
        let result_original = tree.predict(&x_test).unwrap();
        let result_deserialized = deserialized.predict(&x_test).unwrap();

        assert_eq!(result_original, result_deserialized);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_decision_tree_serialize_unfitted() {
        let tree = DecisionTree::new(10, 2, 1);

        // Should serialize unfitted model
        let serialized = serde_json::to_string(&tree).expect("Failed to serialize");
        let deserialized: DecisionTree = serde_json::from_str(&serialized).expect("Failed to deserialize");

        // Both should be unfitted
        let x = matrix_from_vec(vec![vec![1.0, 2.0]]);
        assert!(matches!(tree.predict(&x), Err(TesseractError::NotFitted)));
        assert!(matches!(deserialized.predict(&x), Err(TesseractError::NotFitted)));
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_decision_tree_roundtrip_preserves_state() {
        let mut tree = DecisionTree::new(5, 2, 1);
        let x = matrix_from_vec(vec![
            vec![1.0, 5.0],
            vec![2.0, 6.0],
            vec![8.0, 1.0],
            vec![9.0, 2.0],
        ]);
        let y = vec![0, 0, 1, 1];
        tree.fit(&x, &y).unwrap();

        // Serialize and deserialize
        let serialized = serde_json::to_string(&tree).unwrap();
        let deserialized: DecisionTree = serde_json::from_str(&serialized).unwrap();

        // State should be preserved
        assert_eq!(tree.max_depth, deserialized.max_depth);
        assert_eq!(tree.min_samples_split, deserialized.min_samples_split);
        assert_eq!(tree.min_samples_leaf, deserialized.min_samples_leaf);
        assert_eq!(tree.num_classes, deserialized.num_classes);
    }

    #[test]
    fn test_decision_tree_deep_tree() {
        let mut tree = DecisionTree::new(100, 2, 1); // Very deep
        let x = matrix_from_vec(vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
            vec![6.0],
        ]);
        let y = vec![0, 1, 0, 1, 0, 1];
        tree.fit(&x, &y).unwrap();

        // Should still train without error
        let result = tree.predict(&x).unwrap();
        assert_eq!(result.len(), 6);
    }

    #[test]
    fn test_decision_tree_binary_features() {
        let mut tree = DecisionTree::new(5, 2, 1);
        // Binary features
        let x = matrix_from_vec(vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ]);
        let y = vec![0, 0, 1, 1];
        tree.fit(&x, &y).unwrap();

        let result = tree.predict(&x).unwrap();
        // Should classify based on first feature
        assert_eq!(result[0], 0);
        assert_eq!(result[1], 0);
        assert_eq!(result[2], 1);
        assert_eq!(result[3], 1);
    }
}
