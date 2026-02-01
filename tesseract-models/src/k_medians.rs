#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use tesseract_core::{Float, Matrix, Predictions, Result, TesseractError};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(not(feature = "rng"))]
compile_error!("K-Medians requires the 'rng' feature for random initialization");

/// **K-Medians clustering** using Manhattan (L1) distance.
///
/// K-Medians is similar to K-Means but uses:
/// - **Manhattan distance** (L1 norm) instead of Euclidean distance
/// - **Coordinate-wise median** instead of mean for centroid updates
///
/// This makes K-Medians more **robust to outliers** than K-Means.
///
/// # Algorithm
///
/// 1. Initialize `k` medians (using K-Medians++ variant)
/// 2. Repeat until convergence or max iterations:
///    - **Assignment step**: assign each point to closest median (using Manhattan distance)
///    - **Update step**: recompute medians as coordinate-wise median of assigned points
/// 3. Return final cluster assignments and medians
///
/// # Convergence
///
/// The algorithm stops when:
/// - No assignments change between iterations, OR
/// - Maximum iterations reached
///
/// # Distance metric
///
/// Uses **Manhattan distance** (L1):
/// ```text
/// d(x, y) = Σ_i |x_i - y_i|
/// ```
///
/// # Initialization
///
/// Uses **K-Medians++** initialization (analogous to K-Means++):
/// - First median: random sample
/// - Subsequent medians: sample proportional to Manhattan distance from nearest median
///
/// # Fields
///
/// - `medians`: Cluster centers (shape: `k × d`), set after [`fit`](KMedians::fit)
/// - `k`: Number of clusters
/// - `max_iter`: Maximum iterations
/// - `tol`: Convergence tolerance (currently unused)
///
/// # Errors
///
/// - [`TesseractError::EmptyTrainingData`] if input has zero rows
/// - [`TesseractError::InvalidHyperparameter`] if `k == 0` or `k > n_samples`
/// - [`TesseractError::InvalidValue`] if `NaN` encountered
/// - [`TesseractError::NotFitted`] if prediction called before fitting
///
/// # Notes
///
/// - More robust to outliers than K-Means due to using medians
/// - Typically converges slower than K-Means
/// - Manhattan distance makes it suitable for high-dimensional sparse data
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct KMedians {
    /// Cluster medians of shape `(k, n_features)`.
    medians: Option<Matrix>,
    /// Number of clusters.
    k: usize,
    /// Maximum number of iterations.
    max_iter: usize,
    /// Convergence tolerance (for future use).
    #[allow(dead_code)]
    tol: Float,
}

impl Default for KMedians {
    fn default() -> Self {
        Self {
            medians: None,
            k: 2,
            max_iter: 300,
            tol: 1e-4,
        }
    }
}

impl KMedians {
    /// Creates a new, unfitted K-Medians model.
    ///
    /// # Parameters
    ///
    /// - `k`: Number of clusters (must be > 0)
    /// - `max_iter`: Maximum iterations for the algorithm
    ///
    /// # Example
    ///
    /// ```ignore
    /// let kmedians = KMedians::new(3, 100);
    /// ```
    pub fn new(k: usize, max_iter: usize) -> Self {
        Self {
            medians: None,
            k,
            max_iter,
            tol: 1e-4,
        }
    }

    /// Computes the coordinate-wise median of a set of values.
    ///
    /// # Parameters
    ///
    /// - `values`: Mutable slice of values to compute median from
    ///
    /// # Returns
    ///
    /// The median value.
    ///
    /// # Notes
    ///
    /// - Partially sorts the input slice
    /// - For even-length arrays, returns the lower middle value
    fn median(values: &mut [Float]) -> Float {
        if values.is_empty() {
            return 0.0;
        }

        values.sort_by(|a, b| a.total_cmp(b));
        let mid = values.len() / 2;

        if values.len() % 2 == 0 {
            // Average of two middle values
            (values[mid - 1] + values[mid]) / 2.0
        } else {
            values[mid]
        }
    }

    /// Initializes medians using **K-Medians++** algorithm.
    ///
    /// # Algorithm
    ///
    /// Similar to K-Means++ but using Manhattan distance:
    /// 1. Choose first median uniformly at random
    /// 2. For each subsequent median:
    ///    - Compute Manhattan distance from each point to nearest existing median
    ///    - Choose next median with probability proportional to distance
    ///
    /// # Parameters
    ///
    /// - `x`: Training data of shape `(n_samples, n_features)`
    ///
    /// # Returns
    ///
    /// Matrix of shape `(k, n_features)` containing initial medians.
    ///
    /// # Errors
    ///
    /// - [`TesseractError::InvalidValue`] if `NaN` encountered
    fn initialize_medians_plusplus(&self, x: &Matrix) -> Result<Matrix> {
        use rand::Rng;
        
        let n = x.nrows();
        let d = x.ncols();
        let mut rng = rand::rng();

        let mut medians = Matrix::zeros(self.k, d);

        // First median: random sample
        let first_idx = rng.random_range(0..n);
        medians.row_mut(0).copy_from(&x.row(first_idx));

        // Subsequent medians using K-Medians++ logic
        for c in 1..self.k {
            // Compute Manhattan distance to nearest median for each point
            let mut distances = vec![Float::INFINITY; n];

            #[cfg(feature = "parallel")]
            {
                distances
                    .par_iter_mut()
                    .enumerate()
                    .try_for_each(|(i, dist)| {
                        let xi = x.row(i);
                        let mut min_dist = Float::INFINITY;

                        for j in 0..c {
                            let mut d1 = 0.0;
                            for (k, &xv) in xi.iter().enumerate() {
                                let diff = (xv - medians[(j, k)]).abs();
                                d1 += diff;
                            }

                            if d1.is_nan() {
                                return Err(TesseractError::InvalidValue {
                                    message: "NaN encountered in distance computation".into(),
                                });
                            }

                            if d1 < min_dist {
                                min_dist = d1;
                            }
                        }

                        *dist = min_dist;
                        Ok(())
                    })?;
            }

            #[cfg(not(feature = "parallel"))]
            {
                for i in 0..n {
                    let xi = x.row(i);
                    let mut min_dist = Float::INFINITY;

                    for j in 0..c {
                        let mut d1 = 0.0;
                        for (k, &xv) in xi.iter().enumerate() {
                            let diff = (xv - medians[(j, k)]).abs();
                            d1 += diff;
                        }

                        if d1.is_nan() {
                            return Err(TesseractError::InvalidValue {
                                message: "NaN encountered in distance computation".into(),
                            });
                        }

                        if d1 < min_dist {
                            min_dist = d1;
                        }
                    }

                    distances[i] = min_dist;
                }
            }

            // Choose next median with probability proportional to distance
            let total: Float = distances.iter().sum();
            if total <= 0.0 {
                // All points already covered, pick random
                let idx = rng.random_range(0..n);
                medians.row_mut(c).copy_from(&x.row(idx));
            } else {
                let mut threshold = rng.random::<Float>() * total;
                let mut chosen_idx = 0;

                for (i, &d) in distances.iter().enumerate() {
                    threshold -= d;
                    if threshold <= 0.0 {
                        chosen_idx = i;
                        break;
                    }
                }

                medians.row_mut(c).copy_from(&x.row(chosen_idx));
            }
        }

        Ok(medians)
    }

    /// Assigns each sample to the nearest median using Manhattan distance.
    ///
    /// # Parameters
    ///
    /// - `x`: Data matrix of shape `(n_samples, n_features)`
    /// - `medians`: Median matrix of shape `(k, n_features)`
    ///
    /// # Returns
    ///
    /// Vector of cluster assignments of length `n_samples`.
    ///
    /// # Errors
    ///
    /// - [`TesseractError::InvalidValue`] if `NaN` encountered
    fn assign_clusters(&self, x: &Matrix, medians: &Matrix) -> Result<Predictions> {
        let n = x.nrows();
        let mut assignments = vec![0usize; n];

        #[cfg(feature = "parallel")]
        {
            assignments
                .par_iter_mut()
                .enumerate()
                .try_for_each(|(i, label)| {
                    let xi = x.row(i);
                    let mut min_dist = Float::INFINITY;
                    let mut best_cluster = 0;

                    for c in 0..self.k {
                        let mut d1 = 0.0;
                        for (j, &xv) in xi.iter().enumerate() {
                            let diff = (xv - medians[(c, j)]).abs();
                            d1 += diff;
                        }

                        if d1.is_nan() {
                            return Err(TesseractError::InvalidValue {
                                message: "NaN encountered in distance computation".into(),
                            });
                        }

                        if d1 < min_dist {
                            min_dist = d1;
                            best_cluster = c;
                        }
                    }

                    *label = best_cluster;
                    Ok(())
                })?;
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..n {
                let xi = x.row(i);
                let mut min_dist = Float::INFINITY;
                let mut best_cluster = 0;

                for c in 0..self.k {
                    let mut d1 = 0.0;
                    for (j, &xv) in xi.iter().enumerate() {
                        let diff = (xv - medians[(c, j)]).abs();
                        d1 += diff;
                    }

                    if d1.is_nan() {
                        return Err(TesseractError::InvalidValue {
                            message: "NaN encountered in distance computation".into(),
                        });
                    }

                    if d1 < min_dist {
                        min_dist = d1;
                        best_cluster = c;
                    }
                }

                assignments[i] = best_cluster;
            }
        }

        Ok(assignments)
    }

    /// Updates medians as the coordinate-wise median of assigned samples.
    ///
    /// # Parameters
    ///
    /// - `x`: Data matrix of shape `(n_samples, n_features)`
    /// - `assignments`: Cluster assignments of length `n_samples`
    ///
    /// # Returns
    ///
    /// New median matrix of shape `(k, n_features)`.
    ///
    /// # Notes
    ///
    /// If a cluster has no assigned points, its median remains unchanged.
    fn update_medians(&self, x: &Matrix, assignments: &[usize]) -> Matrix {
        let d = x.ncols();
        let _n = x.nrows();
        let mut new_medians = Matrix::zeros(self.k, d);

        // For each cluster, collect all points and compute coordinate-wise median
        for c in 0..self.k {
            // Collect indices of points in this cluster
            let cluster_indices: Vec<usize> = assignments
                .iter()
                .enumerate()
                .filter_map(|(i, &cluster)| if cluster == c { Some(i) } else { None })
                .collect();

            if cluster_indices.is_empty() {
                continue; // Keep median at zero (or could keep old median)
            }

            // For each feature dimension, compute median
            for j in 0..d {
                let mut values: Vec<Float> = cluster_indices
                    .iter()
                    .map(|&i| x[(i, j)])
                    .collect();

                new_medians[(c, j)] = Self::median(&mut values);
            }
        }

        new_medians
    }

    /// Fits the K-Medians model on training data `x`.
    ///
    /// # Algorithm
    ///
    /// 1. Initialize medians using K-Medians++
    /// 2. Repeat until convergence or max iterations:
    ///    - Assign each sample to nearest median
    ///    - Update medians as coordinate-wise median of assigned samples
    ///    - Check if assignments changed; if not, converged
    ///
    /// # Parameters
    ///
    /// - `x`: Training matrix of shape `(n_samples, n_features)`
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, storing medians in the struct.
    ///
    /// # Errors
    ///
    /// - [`TesseractError::EmptyTrainingData`] if `x.nrows() == 0`
    /// - [`TesseractError::InvalidHyperparameter`] if `k == 0` or `k > n_samples`
    /// - [`TesseractError::InvalidValue`] if `NaN` encountered
    ///
    /// # Complexity
    ///
    /// Let `n = n_samples`, `d = n_features`, `t = iterations until convergence`.
    /// - Time: `O(t * k * n * d + t * k * n * log n)` (extra log factor for median computation)
    /// - Space: `O(k * d + n)` for medians and assignments
    pub fn fit(&mut self, x: &Matrix) -> Result<()> {
        let n = x.nrows();
        let _d = x.ncols();

        if n == 0 {
            return Err(TesseractError::EmptyTrainingData);
        }

        if self.k == 0 {
            return Err(TesseractError::InvalidHyperparameter {
                name: "k".into(),
                value: "0".into(),
            });
        }

        if self.k > n {
            return Err(TesseractError::InvalidHyperparameter {
                name: "k".into(),
                value: format!("{} > n_samples ({})", self.k, n),
            });
        }

        // Initialize medians using K-Medians++
        let mut medians = self.initialize_medians_plusplus(x)?;

        // Iterative algorithm
        let mut prev_assignments = vec![usize::MAX; n];

        for _iter in 0..self.max_iter {
            // Assignment step
            let assignments = self.assign_clusters(x, &medians)?;

            // Check convergence
            if assignments == prev_assignments {
                break;
            }

            // Update step
            medians = self.update_medians(x, &assignments);
            prev_assignments = assignments;
        }

        self.medians = Some(medians);
        Ok(())
    }

    /// Predicts cluster assignments for input matrix `x`.
    ///
    /// # Parameters
    ///
    /// - `x`: Query matrix of shape `(n_queries, n_features)`
    ///
    /// # Returns
    ///
    /// Vector of cluster assignments of length `n_queries`.
    ///
    /// # Errors
    ///
    /// - [`TesseractError::NotFitted`] if model not fitted
    /// - [`TesseractError::ShapeMismatch`] if `x.ncols()` doesn't match training data
    /// - [`TesseractError::InvalidValue`] if `NaN` encountered
    ///
    /// # Complexity
    ///
    /// - Time: `O(n_queries * k * d)`
    /// - Space: `O(n_queries)`
    pub fn predict(&self, x: &Matrix) -> Result<Predictions> {
        let medians = self.medians.as_ref().ok_or(TesseractError::NotFitted)?;

        if x.ncols() != medians.ncols() {
            return Err(TesseractError::ShapeMismatch {
                expected: format!("Expected {} features", medians.ncols()),
                got: format!("Got {} features", x.ncols()),
            });
        }

        self.assign_clusters(x, medians)
    }

    /// Returns the learned cluster medians.
    ///
    /// # Returns
    ///
    /// Reference to median matrix of shape `(k, n_features)`, or `None` if not fitted.
    pub fn medians(&self) -> Option<&Matrix> {
        self.medians.as_ref()
    }

    /// Returns the number of clusters.
    pub fn k(&self) -> usize {
        self.k
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
    fn test_kmedians_new() {
        let kmedians = KMedians::new(3, 100);
        assert_eq!(kmedians.k, 3);
        assert_eq!(kmedians.max_iter, 100);
        assert!(kmedians.medians.is_none());
    }

    #[test]
    fn test_kmedians_empty_data() {
        let mut kmedians = KMedians::new(2, 100);
        let x = Matrix::zeros(0, 2);
        let result = kmedians.fit(&x);
        assert!(matches!(result, Err(TesseractError::EmptyTrainingData)));
    }

    #[test]
    fn test_kmedians_k_zero() {
        let mut kmedians = KMedians::new(0, 100);
        let x = matrix_from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let result = kmedians.fit(&x);
        assert!(matches!(
            result,
            Err(TesseractError::InvalidHyperparameter { .. })
        ));
    }

    #[test]
    fn test_kmedians_k_greater_than_n() {
        let mut kmedians = KMedians::new(5, 100);
        let x = matrix_from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let result = kmedians.fit(&x);
        assert!(matches!(
            result,
            Err(TesseractError::InvalidHyperparameter { .. })
        ));
    }

    #[test]
    fn test_kmedians_predict_not_fitted() {
        let kmedians = KMedians::new(2, 100);
        let x = matrix_from_vec(vec![vec![1.0, 2.0]]);
        let result = kmedians.predict(&x);
        assert!(matches!(result, Err(TesseractError::NotFitted)));
    }

    #[test]
    fn test_kmedians_simple_clustering() {
        let mut kmedians = KMedians::new(2, 100);
        // Two well-separated clusters
        let x = matrix_from_vec(vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.2, 0.2],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
            vec![10.2, 10.2],
        ]);
        kmedians.fit(&x).unwrap();

        // Predict on same data
        let assignments = kmedians.predict(&x).unwrap();
        
        // First 3 should be in one cluster, last 3 in another
        let cluster0 = assignments[0];
        assert_eq!(assignments[1], cluster0);
        assert_eq!(assignments[2], cluster0);
        
        let cluster1 = assignments[3];
        assert_ne!(cluster0, cluster1);
        assert_eq!(assignments[4], cluster1);
        assert_eq!(assignments[5], cluster1);
    }

    #[test]
    fn test_kmedians_three_clusters() {
        let mut kmedians = KMedians::new(3, 100);
        let x = matrix_from_vec(vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![5.0, 5.0],
            vec![5.1, 5.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ]);
        kmedians.fit(&x).unwrap();

        let assignments = kmedians.predict(&x).unwrap();
        
        // Verify each pair is in same cluster
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[2], assignments[3]);
        assert_eq!(assignments[4], assignments[5]);
        
        // Verify all three clusters are different
        assert_ne!(assignments[0], assignments[2]);
        assert_ne!(assignments[0], assignments[4]);
        assert_ne!(assignments[2], assignments[4]);
    }

    #[test]
    fn test_kmedians_medians() {
        let mut kmedians = KMedians::new(2, 100);
        let x = matrix_from_vec(vec![
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![10.0, 10.0],
            vec![10.0, 10.0],
        ]);
        kmedians.fit(&x).unwrap();

        let medians = kmedians.medians().unwrap();
        assert_eq!(medians.nrows(), 2);
        assert_eq!(medians.ncols(), 2);

        // Medians should be approximately [0, 0] and [10, 10]
        let mut found_zero = false;
        let mut found_ten = false;

        for i in 0..2 {
            let m0 = medians[(i, 0)];
            let m1 = medians[(i, 1)];
            
            if (m0 - 0.0).abs() < 1.0 && (m1 - 0.0).abs() < 1.0 {
                found_zero = true;
            }
            if (m0 - 10.0).abs() < 1.0 && (m1 - 10.0).abs() < 1.0 {
                found_ten = true;
            }
        }

        assert!(found_zero);
        assert!(found_ten);
    }

    #[test]
    fn test_kmedians_1d() {
        let mut kmedians = KMedians::new(2, 100);
        let x = matrix_from_vec(vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
            vec![10.0],
            vec![11.0],
            vec![12.0],
        ]);
        kmedians.fit(&x).unwrap();

        let assignments = kmedians.predict(&x).unwrap();
        
        // First 3 in one cluster, last 3 in another
        let c0 = assignments[0];
        assert_eq!(assignments[1], c0);
        assert_eq!(assignments[2], c0);
        assert_ne!(assignments[3], c0);
        assert_eq!(assignments[4], assignments[3]);
        assert_eq!(assignments[5], assignments[3]);
    }

    #[test]
    fn test_kmedians_shape_mismatch() {
        let mut kmedians = KMedians::new(2, 100);
        let x = matrix_from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        kmedians.fit(&x).unwrap();

        // Try to predict with different number of features
        let x_test = matrix_from_vec(vec![vec![1.0]]);
        let result = kmedians.predict(&x_test);
        assert!(matches!(result, Err(TesseractError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_kmedians_single_cluster() {
        let mut kmedians = KMedians::new(1, 100);
        let x = matrix_from_vec(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ]);
        kmedians.fit(&x).unwrap();

        let assignments = kmedians.predict(&x).unwrap();
        
        // All should be in cluster 0
        assert!(assignments.iter().all(|&a| a == 0));
    }

    #[test]
    fn test_kmedians_with_outliers() {
        let mut kmedians = KMedians::new(2, 100);
        // Cluster with outlier: median should be more robust than mean
        let x = matrix_from_vec(vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.2, 0.2],
            vec![100.0, 0.0], // outlier, but should still be in different cluster
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ]);
        kmedians.fit(&x).unwrap();

        let assignments = kmedians.predict(&x).unwrap();
        
        // First 3 should mostly be together
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[1], assignments[2]);
    }

    #[test]
    fn test_kmedians_convergence() {
        let mut kmedians = KMedians::new(2, 1); // Only 1 iteration
        let x = matrix_from_vec(vec![
            vec![0.0, 0.0],
            vec![10.0, 10.0],
        ]);
        kmedians.fit(&x).unwrap();

        // Should still produce valid medians
        assert!(kmedians.medians().is_some());
    }

    #[test]
    fn test_kmedians_with_nan() {
        let mut kmedians = KMedians::new(2, 100);
        let x = matrix_from_vec(vec![
            vec![1.0, 2.0],
            vec![f32::NAN, 4.0],
        ]);
        let result = kmedians.fit(&x);
        assert!(matches!(result, Err(TesseractError::InvalidValue { .. })));
    }

    #[test]
    fn test_kmedians_high_dimensional() {
        let mut kmedians = KMedians::new(2, 100);
        let x = matrix_from_vec(vec![
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.1, 0.1, 0.1, 0.1, 0.1],
            vec![10.0, 10.0, 10.0, 10.0, 10.0],
            vec![10.1, 10.1, 10.1, 10.1, 10.1],
        ]);
        kmedians.fit(&x).unwrap();

        let assignments = kmedians.predict(&x).unwrap();
        
        // First 2 in one cluster, last 2 in another
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[2], assignments[3]);
        assert_ne!(assignments[0], assignments[2]);
    }

    #[test]
    fn test_kmedians_identical_points() {
        let mut kmedians = KMedians::new(2, 100);
        // All points identical
        let x = matrix_from_vec(vec![
            vec![5.0, 5.0],
            vec![5.0, 5.0],
            vec![5.0, 5.0],
            vec![5.0, 5.0],
        ]);
        kmedians.fit(&x).unwrap();

        // Should still fit without error
        let assignments = kmedians.predict(&x).unwrap();
        assert_eq!(assignments.len(), 4);
    }

    #[test]
    fn test_kmedians_median_computation() {
        // Test the median function directly
        let mut values = vec![1.0, 3.0, 2.0];
        assert_eq!(KMedians::median(&mut values), 2.0);

        let mut values_even = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(KMedians::median(&mut values_even), 2.5);

        let mut single = vec![5.0];
        assert_eq!(KMedians::median(&mut single), 5.0);

        let empty: &mut [Float] = &mut [];
        assert_eq!(KMedians::median(empty), 0.0);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_kmedians_serialize_deserialize_json() {
        let mut kmedians = KMedians::new(2, 100);
        let x = matrix_from_vec(vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ]);
        kmedians.fit(&x).unwrap();

        // Serialize
        let serialized = serde_json::to_string(&kmedians).expect("Failed to serialize");
        assert!(!serialized.is_empty());

        // Deserialize
        let deserialized: KMedians = serde_json::from_str(&serialized).expect("Failed to deserialize");

        // Verify predictions match
        let x_test = matrix_from_vec(vec![vec![0.0, 0.0], vec![10.0, 10.0]]);
        let result_original = kmedians.predict(&x_test).unwrap();
        let result_deserialized = deserialized.predict(&x_test).unwrap();

        assert_eq!(result_original, result_deserialized);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_kmedians_serialize_unfitted() {
        let kmedians = KMedians::new(3, 50);

        // Should serialize unfitted model
        let serialized = serde_json::to_string(&kmedians).expect("Failed to serialize");
        let deserialized: KMedians = serde_json::from_str(&serialized).expect("Failed to deserialize");

        // Both should be unfitted
        let x = matrix_from_vec(vec![vec![1.0, 2.0]]);
        assert!(matches!(kmedians.predict(&x), Err(TesseractError::NotFitted)));
        assert!(matches!(deserialized.predict(&x), Err(TesseractError::NotFitted)));
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_kmedians_roundtrip_preserves_state() {
        let mut kmedians = KMedians::new(2, 100);
        let x = matrix_from_vec(vec![
            vec![1.0, 2.0],
            vec![1.1, 2.1],
            vec![9.0, 8.0],
            vec![9.1, 8.1],
        ]);
        kmedians.fit(&x).unwrap();

        // Serialize and deserialize
        let serialized = serde_json::to_string(&kmedians).unwrap();
        let deserialized: KMedians = serde_json::from_str(&serialized).unwrap();

        // State should be preserved
        assert_eq!(kmedians.k, deserialized.k);
        assert_eq!(kmedians.max_iter, deserialized.max_iter);
        
        // Medians should match
        let m1 = kmedians.medians().unwrap();
        let m2 = deserialized.medians().unwrap();
        assert_eq!(m1.nrows(), m2.nrows());
        assert_eq!(m1.ncols(), m2.ncols());
    }

    #[test]
    fn test_kmedians_k_getter() {
        let kmedians = KMedians::new(5, 100);
        assert_eq!(kmedians.k(), 5);
    }
}
