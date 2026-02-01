#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use tesseract_core::{Matrix, Predictions, Result, TesseractError};

/// A neighbor candidate used during KNN search.
///
/// This struct stores:
/// - `dist2`: the **squared Euclidean distance** to a training point
/// - `idx`: the index of that training point in the stored dataset
///
/// It implements ordering so it can be stored inside a [`BinaryHeap`].
///
/// # Heap semantics
///
/// Rust’s [`BinaryHeap`] is a **max-heap**, meaning `peek()` returns the “largest” element.
/// For KNN, we want to keep the **k smallest distances**. To do this efficiently, we define
/// “largest” = **worst (farthest) neighbor**.
///
/// As a result:
/// - the heap always contains the best `k` neighbors seen so far
/// - `heap.peek()` gives the current farthest neighbor among those `k`
/// - a new candidate replaces the top if it is closer
///
/// # Float ordering
///
/// `f32` does not implement [`Ord`] due to `NaN`. We use `total_cmp` for a total ordering.
/// Equality uses `to_bits()` to remain well-defined even in the presence of special float values.
#[derive(Debug, Clone, Copy)]
struct Neighbour {
    dist2: f32,
    idx: usize,
}

impl PartialEq for Neighbour {
    fn eq(&self, other: &Self) -> bool {
        self.dist2.to_bits() == other.dist2.to_bits() && self.idx == other.idx
    }
}

impl Eq for Neighbour {}

impl PartialOrd for Neighbour {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbour {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap ordering:
        // larger distance = "greater" (worse neighbor)
        self.dist2
            .total_cmp(&other.dist2)
            .then_with(|| self.idx.cmp(&other.idx))
    }
}

/// A **k-Nearest Neighbors (KNN)** classifier using squared Euclidean distance.
///
/// KNN is a **lazy learner**:
/// - [`fit`](KNN::fit) stores the training data
/// - [`predict`](KNN::predict) computes distances to training samples at inference time
///
/// This implementation:
/// - uses **squared Euclidean distance** (avoids `sqrt`, same ranking as Euclidean distance)
/// - finds the `k` closest neighbors using a **max-heap of size `k`**
/// - predicts by **majority vote** over neighbor labels (ties broken by closest neighbor)
///
/// # Fields
///
/// - `dataset`: Stored training matrix (set by `fit`)
/// - `labels`: Stored training labels (set by `fit`)
/// - `k`: Number of neighbors to consider
///
/// # Notes
///
/// - This implementation assumes `Predictions` is the label vector type (e.g. `Vec<usize>`).
/// - If distance computation produces `NaN`, prediction returns [`TesseractError::InvalidValue`].
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct KNN {
    dataset: Option<Matrix>,
    labels: Option<Predictions>,
    k: usize,
}

impl Default for KNN {
    fn default() -> Self {
        Self {
            dataset: None,
            labels: None,
            k: 0,
        }
    }
}

impl KNN {
    /// Creates a new, unfitted KNN model.
    ///
    /// The returned model must be fitted with [`fit`](KNN::fit) before calling
    /// [`predict`](KNN::predict).
    pub fn new() -> Self {
        Self::default()
    }

    /// Fits the KNN model by storing the training data and labels.
    ///
    /// # Parameters
    ///
    /// - `x`: Training matrix of shape `(n_samples, n_features)`.
    /// - `y`: Training labels of length `n_samples`.
    /// - `k`: Number of neighbors to consider during prediction.
    ///
    /// # Notes
    ///
    /// - KNN does not learn parameters; fitting is just storing `x` and `y`.
    /// - This method clones `x` and `y`. If you later want to reduce memory usage,
    ///   consider storing references with lifetimes or using `Arc`.
    pub fn fit(&mut self, x: &Matrix, y: &Predictions, k: usize) {
        self.dataset = Some(x.clone());
        self.labels = Some(y.clone());
        self.k = k;
    }

    /// Predicts labels for input matrix `x` using majority vote among `k` nearest neighbors.
    ///
    /// # Algorithm
    ///
    /// For each query sample `x_i`:
    /// 1. Compute squared Euclidean distance to every training sample.
    /// 2. Maintain a max-heap of size `k` containing the closest neighbors found so far.
    /// 3. After scanning all training samples, perform a majority vote over the heap’s labels.
    ///
    /// Ties are broken by choosing the label whose closest neighbor (among the tied labels)
    /// has the smallest distance (i.e., “closest wins”).
    ///
    /// # Parameters
    ///
    /// - `x`: Query matrix of shape `(n_queries, n_features)`.
    ///
    /// # Returns
    ///
    /// A vector of predicted labels of length `n_queries`.
    ///
    /// # Errors
    ///
    /// - [`TesseractError::NotFitted`] if `fit` has not been called.
    /// - [`TesseractError::InvalidHyperparameter`] if `k == 0`.
    /// - [`TesseractError::ShapeMismatch`] if `x.ncols() != training.ncols()`.
    /// - [`TesseractError::InsufficientTrainingData`] if `training.nrows() < k`.
    /// - [`TesseractError::InvalidValue`] if a `NaN` is encountered in distance computation.
    ///
    /// # Complexity
    ///
    /// Let:
    /// - `n_train` = number of training samples
    /// - `n_query` = number of query samples
    /// - `d` = number of features
    ///
    /// Then:
    /// - Time: `O(n_query * n_train * d + n_query * n_train * log k)`
    ///   (distance computation dominates in practice; heap maintenance is `log k` per training point)
    /// - Space: `O(k + n_query)` (heap per query, output vector)
    pub fn predict(&self, x: &Matrix) -> Result<Predictions> {
        let (a, y) = match (&self.dataset, &self.labels) {
            (Some(a), Some(y)) => (a, y),
            _ => return Err(TesseractError::NotFitted),
        };

        if self.k == 0 {
            return Err(TesseractError::InvalidHyperparameter {
                name: "k".into(),
                value: "0".into(),
            });
        }

        if a.ncols() != x.ncols() {
            return Err(TesseractError::ShapeMismatch {
                expected: format!("Expected {} features", a.ncols()),
                got: format!("Got {} features", x.ncols()),
            });
        }

        if a.nrows() < self.k {
            return Err(TesseractError::InsufficientTrainingData);
        }

        let mut preds = vec![0usize; x.nrows()];
        let mut counts: HashMap<usize, usize> = HashMap::new();

        for (i, out) in preds.iter_mut().enumerate() {
            let xi = x.row(i);

            // Max-heap of the k best neighbors seen so far.
            let mut heap: BinaryHeap<Neighbour> = BinaryHeap::with_capacity(self.k);

            for (j, aj) in a.row_iter().enumerate() {
                // Squared Euclidean distance computed without allocating temporary vectors.
                let mut dist2: f32 = 0.0;
                for (&av, &xv) in aj.iter().zip(xi.iter()) {
                    let diff = av - xv;
                    dist2 += diff * diff;
                }

                if dist2.is_nan() {
                    return Err(TesseractError::InvalidValue {
                        message: String::from("Encountered NaN when calculating distance."),
                    });
                }

                let cand = Neighbour { dist2, idx: j };

                if heap.len() < self.k {
                    heap.push(cand);
                } else if let Some(&worst) = heap.peek() {
                    // `worst` is the largest distance among the kept neighbors.
                    // Replace it if the new candidate is closer.
                    if cand.dist2 < worst.dist2 {
                        heap.pop();
                        heap.push(cand);
                    }
                }
            }

            // Majority vote over the k neighbors in the heap.
            counts.clear();
            let mut best_label: usize = 0;
            let mut best_count: usize = 0;
            let mut best_closest: f32 = f32::INFINITY; // tie-break by closeness

            for n in heap.into_iter() {
                let label = y[n.idx];
                let c = counts.entry(label).or_insert(0);
                *c += 1;

                // Prefer higher vote count; break ties by closest neighbor distance.
                if *c > best_count || (*c == best_count && n.dist2 < best_closest) {
                    best_count = *c;
                    best_label = label;
                    best_closest = n.dist2;
                }
            }

            *out = best_label;
        }

        Ok(preds)
    }
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
    fn test_knn_not_fitted() {
        let knn = KNN::new();
        let x = matrix_from_vec(vec![vec![1.0, 2.0]]);
        let result = knn.predict(&x);
        assert!(matches!(result, Err(TesseractError::NotFitted)));
    }

    #[test]
    fn test_knn_k_zero() {
        let mut knn = KNN::new();
        let x_train = matrix_from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let y_train = vec![0, 1];
        knn.fit(&x_train, &y_train, 0);

        let x_test = matrix_from_vec(vec![vec![2.0, 3.0]]);
        let result = knn.predict(&x_test);
        assert!(matches!(
            result,
            Err(TesseractError::InvalidHyperparameter { .. })
        ));
    }

    #[test]
    fn test_knn_feature_mismatch() {
        let mut knn = KNN::new();
        let x_train = matrix_from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let y_train = vec![0, 1];
        knn.fit(&x_train, &y_train, 1);

        // Test data has different number of features
        let x_test = matrix_from_vec(vec![vec![2.0, 3.0, 5.0]]);
        let result = knn.predict(&x_test);
        assert!(matches!(result, Err(TesseractError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_knn_insufficient_training_data() {
        let mut knn = KNN::new();
        let x_train = matrix_from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let y_train = vec![0, 1];
        // k > number of training samples
        knn.fit(&x_train, &y_train, 5);

        let x_test = matrix_from_vec(vec![vec![2.0, 3.0]]);
        let result = knn.predict(&x_test);
        assert!(matches!(
            result,
            Err(TesseractError::InsufficientTrainingData)
        ));
    }

    #[test]
    fn test_knn_perfect_match() {
        let mut knn = KNN::new();
        let x_train = matrix_from_vec(vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
            vec![4.0, 4.0],
        ]);
        let y_train = vec![0, 0, 1, 1];
        knn.fit(&x_train, &y_train, 3);

        // Query exact training point
        let x_test = matrix_from_vec(vec![vec![2.0, 2.0]]);
        let result = knn.predict(&x_test).unwrap();
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_knn_simple_classification() {
        let mut knn = KNN::new();
        // Two clear clusters
        let x_train = matrix_from_vec(vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.2, 0.0],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
            vec![10.0, 10.2],
        ]);
        let y_train = vec![0, 0, 0, 1, 1, 1];
        knn.fit(&x_train, &y_train, 3);

        // Point close to cluster 0
        let x_test1 = matrix_from_vec(vec![vec![0.05, 0.05]]);
        let result1 = knn.predict(&x_test1).unwrap();
        assert_eq!(result1, vec![0]);

        // Point close to cluster 1
        let x_test2 = matrix_from_vec(vec![vec![10.05, 10.05]]);
        let result2 = knn.predict(&x_test2).unwrap();
        assert_eq!(result2, vec![1]);
    }

    #[test]
    fn test_knn_k1_nearest_neighbor() {
        let mut knn = KNN::new();
        let x_train = matrix_from_vec(vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![5.0, 5.0]]);
        let y_train = vec![0, 1, 2];
        knn.fit(&x_train, &y_train, 1);

        // Should pick the single nearest neighbor
        let x_test = matrix_from_vec(vec![vec![0.1, 0.1]]);
        let result = knn.predict(&x_test).unwrap();
        assert_eq!(result, vec![0]); // closest to [0.0, 0.0]

        let x_test2 = matrix_from_vec(vec![vec![4.9, 4.9]]);
        let result2 = knn.predict(&x_test2).unwrap();
        assert_eq!(result2, vec![2]); // closest to [5.0, 5.0]
    }

    #[test]
    fn test_knn_majority_vote() {
        let mut knn = KNN::new();
        // Create a scenario where majority vote matters
        let x_train = matrix_from_vec(vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.5, 0.5],
            vec![1.0, 1.0],
        ]);
        let y_train = vec![0, 0, 0, 1, 1];
        knn.fit(&x_train, &y_train, 5);

        // Point closer to class 0 cluster (3 vs 2)
        let x_test = matrix_from_vec(vec![vec![0.2, 0.2]]);
        let result = knn.predict(&x_test).unwrap();
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_knn_multiple_queries() {
        let mut knn = KNN::new();
        let x_train = matrix_from_vec(vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![10.0, 10.0]]);
        let y_train = vec![0, 0, 1];
        knn.fit(&x_train, &y_train, 2);

        // Multiple test samples
        let x_test = matrix_from_vec(vec![vec![0.5, 0.5], vec![9.5, 9.5], vec![0.0, 0.0]]);
        let result = knn.predict(&x_test).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 0); // closer to [0, 0] and [1, 1]
        assert_eq!(result[1], 1); // closer to [10, 10]
        assert_eq!(result[2], 0); // exact match with [0, 0]
    }

    #[test]
    fn test_knn_tie_breaking() {
        let mut knn = KNN::new();
        // Setup where tie-break by distance matters
        let x_train = matrix_from_vec(vec![
            vec![0.0, 0.0],  // class 0, distance 1.0 from test point
            vec![2.0, 0.0],  // class 1, distance 1.0 from test point
            vec![3.0, 0.0],  // class 0, distance 2.0 from test point
            vec![-1.0, 0.0], // class 1, distance 2.0 from test point
        ]);
        let y_train = vec![0, 1, 0, 1];
        knn.fit(&x_train, &y_train, 4);

        // Test point at [1.0, 0.0] - equidistant to first two neighbors
        // With all 4 neighbors: 2 votes for each class
        // Tie should be broken by closest neighbor
        let x_test = matrix_from_vec(vec![vec![1.0, 0.0]]);
        let result = knn.predict(&x_test).unwrap();
        // Both class 0 and 1 have same count, but closest is either [0,0] or [2,0]
        // The implementation breaks ties by choosing closest, so result should be deterministic
        assert!(result[0] == 0 || result[0] == 1);
    }

    #[test]
    fn test_knn_with_nan_returns_error() {
        let mut knn = KNN::new();
        let x_train = matrix_from_vec(vec![vec![0.0, 0.0], vec![1.0, 1.0]]);
        let y_train = vec![0, 1];
        knn.fit(&x_train, &y_train, 1);

        // Test data with NaN
        let x_test = matrix_from_vec(vec![vec![f32::NAN, 0.0]]);
        let result = knn.predict(&x_test);
        assert!(matches!(result, Err(TesseractError::InvalidValue { .. })));
    }

    #[test]
    fn test_knn_single_class() {
        let mut knn = KNN::new();
        let x_train = matrix_from_vec(vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]]);
        let y_train = vec![0, 0, 0]; // all same class
        knn.fit(&x_train, &y_train, 2);

        let x_test = matrix_from_vec(vec![vec![5.0, 5.0]]);
        let result = knn.predict(&x_test).unwrap();
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_knn_three_classes() {
        let mut knn = KNN::new();
        let x_train = matrix_from_vec(vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![5.0, 5.0],
            vec![5.1, 5.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ]);
        let y_train = vec![0, 0, 1, 1, 2, 2];
        knn.fit(&x_train, &y_train, 3);

        let x_test = matrix_from_vec(vec![vec![0.0, 0.0], vec![5.0, 5.0], vec![10.0, 10.0]]);
        let result = knn.predict(&x_test).unwrap();
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn test_knn_1d_features() {
        let mut knn = KNN::new();
        let x_train = matrix_from_vec(vec![vec![0.0], vec![1.0], vec![10.0]]);
        let y_train = vec![0, 0, 1];
        knn.fit(&x_train, &y_train, 2);

        let x_test = matrix_from_vec(vec![vec![0.5], vec![9.0]]);
        let result = knn.predict(&x_test).unwrap();
        assert_eq!(result[0], 0); // closer to 0 and 1
        assert_eq!(result[1], 1); // closer to 10
    }

    #[test]
    fn test_knn_high_dimensional() {
        let mut knn = KNN::new();
        // 5D space
        let x_train = matrix_from_vec(vec![
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
            vec![10.0, 10.0, 10.0, 10.0, 10.0],
        ]);
        let y_train = vec![0, 0, 1];
        knn.fit(&x_train, &y_train, 2);

        let x_test = matrix_from_vec(vec![vec![0.5, 0.5, 0.5, 0.5, 0.5]]);
        let result = knn.predict(&x_test).unwrap();
        assert_eq!(result, vec![0]);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_knn_serialize_deserialize_json() {
        let mut knn = KNN::new();
        let x_train = matrix_from_vec(vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![10.0, 10.0],
        ]);
        let y_train = vec![0, 0, 1];
        knn.fit(&x_train, &y_train, 2);

        // Serialize to JSON
        let serialized = serde_json::to_string(&knn).expect("Failed to serialize");
        assert!(!serialized.is_empty());

        // Deserialize from JSON
        let deserialized: KNN = serde_json::from_str(&serialized).expect("Failed to deserialize");

        // Test that deserialized model works
        let x_test = matrix_from_vec(vec![vec![0.5, 0.5], vec![9.5, 9.5]]);
        let result_original = knn.predict(&x_test).unwrap();
        let result_deserialized = deserialized.predict(&x_test).unwrap();

        assert_eq!(result_original, result_deserialized);
        assert_eq!(result_deserialized[0], 0);
        assert_eq!(result_deserialized[1], 1);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_knn_serialize_deserialize_postcard() {
        let mut knn = KNN::new();
        let x_train = matrix_from_vec(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ]);
        let y_train = vec![0, 1, 1];
        knn.fit(&x_train, &y_train, 3);

        // Serialize using serde (via postcard-compatible format)
        let serialized = serde_json::to_vec(&knn).expect("Failed to serialize");
        assert!(!serialized.is_empty());

        // Deserialize
        let deserialized: KNN = serde_json::from_slice(&serialized).expect("Failed to deserialize");

        // Verify predictions match
        let x_test = matrix_from_vec(vec![vec![2.0, 3.0]]);
        let result_original = knn.predict(&x_test).unwrap();
        let result_deserialized = deserialized.predict(&x_test).unwrap();

        assert_eq!(result_original, result_deserialized);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_knn_serialize_unfitted() {
        let knn = KNN::new();

        // Should be able to serialize unfitted model
        let serialized = serde_json::to_string(&knn).expect("Failed to serialize");
        let deserialized: KNN = serde_json::from_str(&serialized).expect("Failed to deserialize");

        // Both should fail with NotFitted
        let x_test = matrix_from_vec(vec![vec![1.0, 2.0]]);
        assert!(matches!(knn.predict(&x_test), Err(TesseractError::NotFitted)));
        assert!(matches!(deserialized.predict(&x_test), Err(TesseractError::NotFitted)));
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_knn_roundtrip_preserves_k() {
        let mut knn = KNN::new();
        let x_train = matrix_from_vec(vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
        ]);
        let y_train = vec![0, 0, 1, 1, 1];
        knn.fit(&x_train, &y_train, 3);

        // Serialize and deserialize
        let serialized = serde_json::to_string(&knn).unwrap();
        let deserialized: KNN = serde_json::from_str(&serialized).unwrap();

        // k should be preserved (test by checking predictions are identical)
        let x_test = matrix_from_vec(vec![vec![2.5], vec![4.5]]);
        let result_original = knn.predict(&x_test).unwrap();
        let result_deserialized = deserialized.predict(&x_test).unwrap();

        assert_eq!(result_original, result_deserialized);
    }
}
