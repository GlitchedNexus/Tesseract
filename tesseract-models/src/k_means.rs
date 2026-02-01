#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use tesseract_core::{Float, Matrix, Predictions, Result, TesseractError};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(not(feature = "rng"))]
compile_error!("K-Means requires the 'rng' feature for random initialization");

/// **K-Means clustering** using squared Euclidean distance.
///
/// K-Means is an **unsupervised learning** algorithm that partitions `n` samples
/// into `k` clusters by iteratively:
/// 1. Assigning each sample to the nearest centroid
/// 2. Updating centroids as the mean of assigned samples
///
/// # Algorithm
///
/// **Lloyd's algorithm** (standard K-Means):
/// 1. Initialize `k` centroids (using K-Means++ by default)
/// 2. Repeat until convergence or max iterations:
///    - **Assignment step**: assign each point to closest centroid
///    - **Update step**: recompute centroids as mean of assigned points
/// 3. Return final cluster assignments and centroids
///
/// # Convergence
///
/// The algorithm stops when:
/// - No assignments change between iterations, OR
/// - Maximum iterations reached
///
/// # Distance metric
///
/// Uses **squared Euclidean distance** (avoids `sqrt`, same ranking as Euclidean).
///
/// # Initialization
///
/// Uses **K-Means++** initialization for better convergence:
/// - First centroid: random sample
/// - Subsequent centroids: sample proportional to squared distance from nearest centroid
///
/// # Fields
///
/// - `centroids`: Cluster centers (shape: `k × d`), set after [`fit`](KMeans::fit)
/// - `k`: Number of clusters
/// - `max_iter`: Maximum iterations for Lloyd's algorithm
/// - `tol`: Convergence tolerance (currently unused, stops on stable assignments)
///
/// # Errors
///
/// - [`TesseractError::EmptyTrainingData`] if input has zero rows
/// - [`TesseractError::InvalidHyperparameter`] if `k == 0` or `k > n_samples`
/// - [`TesseractError::InvalidValue`] if `NaN` encountered in distance computation
/// - [`TesseractError::NotFitted`] if prediction called before fitting
///
/// # Notes
///
/// - K-Means assumes **spherical clusters** of similar size
/// - Sensitive to initialization (K-Means++ helps but doesn't guarantee global optimum)
/// - Use multiple random restarts for better results (not yet implemented)
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct KMeans {
    /// Cluster centroids of shape `(k, n_features)`.
    centroids: Option<Matrix>,
    /// Number of clusters.
    k: usize,
    /// Maximum number of iterations.
    max_iter: usize,
    /// Convergence tolerance (for future use).
    #[allow(dead_code)]
    tol: Float,
}

impl Default for KMeans {
    fn default() -> Self {
        Self {
            centroids: None,
            k: 2,
            max_iter: 300,
            tol: 1e-4,
        }
    }
}

impl KMeans {
    /// Creates a new, unfitted K-Means model.
    ///
    /// # Parameters
    ///
    /// - `k`: Number of clusters (must be > 0)
    /// - `max_iter`: Maximum iterations for Lloyd's algorithm
    ///
    /// # Example
    ///
    /// ```ignore
    /// let kmeans = KMeans::new(3, 100);
    /// ```
    pub fn new(k: usize, max_iter: usize) -> Self {
        Self {
            centroids: None,
            k,
            max_iter,
            tol: 1e-4,
        }
    }

    /// Initializes centroids using **K-Means++** algorithm.
    ///
    /// # Algorithm
    ///
    /// 1. Choose first centroid uniformly at random from data points
    /// 2. For each subsequent centroid:
    ///    - Compute squared distance from each point to nearest existing centroid
    ///    - Choose next centroid with probability proportional to squared distance
    ///
    /// This spreads out initial centroids, leading to better convergence.
    ///
    /// # Parameters
    ///
    /// - `x`: Training data of shape `(n_samples, n_features)`
    ///
    /// # Returns
    ///
    /// Matrix of shape `(k, n_features)` containing initial centroids.
    ///
    /// # Errors
    ///
    /// - [`TesseractError::InvalidValue`] if `NaN` encountered
    fn initialize_centroids_plusplus(&self, x: &Matrix) -> Result<Matrix> {
        use rand::Rng;
        
        let n = x.nrows();
        let d = x.ncols();
        let mut rng = rand::rng();

        let mut centroids = Matrix::zeros(self.k, d);

        // First centroid: random sample
        let first_idx = rng.random_range(0..n);
        centroids.row_mut(0).copy_from(&x.row(first_idx));

        // Subsequent centroids using K-Means++ logic
        for c in 1..self.k {
            // Compute squared distance to nearest centroid for each point
            let mut distances = vec![Float::INFINITY; n];

            #[cfg(feature = "parallel")]
            {
                distances
                    .par_iter_mut()
                    .enumerate()
                    .try_for_each(|(i, dist)| {
                        let xi = x.row(i);
                        let mut min_dist2 = Float::INFINITY;

                        for j in 0..c {
                            let mut d2 = 0.0;
                            for (k, &xv) in xi.iter().enumerate() {
                                let diff = xv - centroids[(j, k)];
                                d2 += diff * diff;
                            }

                            if d2.is_nan() {
                                return Err(TesseractError::InvalidValue {
                                    message: "NaN encountered in distance computation".into(),
                                });
                            }

                            if d2 < min_dist2 {
                                min_dist2 = d2;
                            }
                        }

                        *dist = min_dist2;
                        Ok(())
                    })?;
            }

            #[cfg(not(feature = "parallel"))]
            {
                for i in 0..n {
                    let xi = x.row(i);
                    let mut min_dist2 = Float::INFINITY;

                    for j in 0..c {
                        let mut d2 = 0.0;
                        for (k, &xv) in xi.iter().enumerate() {
                            let diff = xv - centroids[(j, k)];
                            d2 += diff * diff;
                        }

                        if d2.is_nan() {
                            return Err(TesseractError::InvalidValue {
                                message: "NaN encountered in distance computation".into(),
                            });
                        }

                        if d2 < min_dist2 {
                            min_dist2 = d2;
                        }
                    }

                    distances[i] = min_dist2;
                }
            }

            // Choose next centroid with probability proportional to squared distance
            let total: Float = distances.iter().sum();
            if total <= 0.0 {
                // All points already covered by existing centroids, just pick random
                let idx = rng.random_range(0..n);
                centroids.row_mut(c).copy_from(&x.row(idx));
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

                centroids.row_mut(c).copy_from(&x.row(chosen_idx));
            }
        }

        Ok(centroids)
    }

    /// Assigns each sample to the nearest centroid.
    ///
    /// # Parameters
    ///
    /// - `x`: Data matrix of shape `(n_samples, n_features)`
    /// - `centroids`: Centroid matrix of shape `(k, n_features)`
    ///
    /// # Returns
    ///
    /// Vector of cluster assignments of length `n_samples`, where each element
    /// is in `0..k`.
    ///
    /// # Errors
    ///
    /// - [`TesseractError::InvalidValue`] if `NaN` encountered
    fn assign_clusters(&self, x: &Matrix, centroids: &Matrix) -> Result<Predictions> {
        let n = x.nrows();
        let mut assignments = vec![0usize; n];

        #[cfg(feature = "parallel")]
        {
            assignments
                .par_iter_mut()
                .enumerate()
                .try_for_each(|(i, label)| {
                    let xi = x.row(i);
                    let mut min_dist2 = Float::INFINITY;
                    let mut best_cluster = 0;

                    for c in 0..self.k {
                        let mut d2 = 0.0;
                        for (j, &xv) in xi.iter().enumerate() {
                            let diff = xv - centroids[(c, j)];
                            d2 += diff * diff;
                        }

                        if d2.is_nan() {
                            return Err(TesseractError::InvalidValue {
                                message: "NaN encountered in distance computation".into(),
                            });
                        }

                        if d2 < min_dist2 {
                            min_dist2 = d2;
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
                let mut min_dist2 = Float::INFINITY;
                let mut best_cluster = 0;

                for c in 0..self.k {
                    let mut d2 = 0.0;
                    for (j, &xv) in xi.iter().enumerate() {
                        let diff = xv - centroids[(c, j)];
                        d2 += diff * diff;
                    }

                    if d2.is_nan() {
                        return Err(TesseractError::InvalidValue {
                            message: "NaN encountered in distance computation".into(),
                        });
                    }

                    if d2 < min_dist2 {
                        min_dist2 = d2;
                        best_cluster = c;
                    }
                }

                assignments[i] = best_cluster;
            }
        }

        Ok(assignments)
    }

    /// Updates centroids as the mean of assigned samples.
    ///
    /// # Parameters
    ///
    /// - `x`: Data matrix of shape `(n_samples, n_features)`
    /// - `assignments`: Cluster assignments of length `n_samples`
    ///
    /// # Returns
    ///
    /// New centroid matrix of shape `(k, n_features)`.
    ///
    /// # Notes
    ///
    /// If a cluster has no assigned points, its centroid remains unchanged.
    fn update_centroids(&self, x: &Matrix, assignments: &[usize]) -> Matrix {
        let d = x.ncols();
        let mut new_centroids = Matrix::zeros(self.k, d);
        let mut counts = vec![0usize; self.k];

        // Sum up all points assigned to each cluster
        for (i, &cluster) in assignments.iter().enumerate() {
            let xi = x.row(i);
            for j in 0..d {
                new_centroids[(cluster, j)] += xi[j];
            }
            counts[cluster] += 1;
        }

        // Divide by count to get mean
        for c in 0..self.k {
            if counts[c] > 0 {
                let count_f = counts[c] as Float;
                for j in 0..d {
                    new_centroids[(c, j)] /= count_f;
                }
            }
            // If count is 0, centroid stays at zero (or could keep old centroid)
        }

        new_centroids
    }

    /// Fits the K-Means model on training data `x`.
    ///
    /// # Algorithm
    ///
    /// 1. Initialize centroids using K-Means++
    /// 2. Repeat until convergence or max iterations:
    ///    - Assign each sample to nearest centroid
    ///    - Update centroids as mean of assigned samples
    ///    - Check if assignments changed; if not, converged
    ///
    /// # Parameters
    ///
    /// - `x`: Training matrix of shape `(n_samples, n_features)`
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, storing centroids in the struct.
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
    /// - Time: `O(t * k * n * d)` per iteration
    /// - Space: `O(k * d + n)` for centroids and assignments
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

        // Initialize centroids using K-Means++
        let mut centroids = self.initialize_centroids_plusplus(x)?;

        // Lloyd's algorithm
        let mut prev_assignments = vec![usize::MAX; n];

        for _iter in 0..self.max_iter {
            // Assignment step
            let assignments = self.assign_clusters(x, &centroids)?;

            // Check convergence
            if assignments == prev_assignments {
                break;
            }

            // Update step
            centroids = self.update_centroids(x, &assignments);
            prev_assignments = assignments;
        }

        self.centroids = Some(centroids);
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
    /// Vector of cluster assignments of length `n_queries`, where each element
    /// is in `0..k`.
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
        let centroids = self.centroids.as_ref().ok_or(TesseractError::NotFitted)?;

        if x.ncols() != centroids.ncols() {
            return Err(TesseractError::ShapeMismatch {
                expected: format!("Expected {} features", centroids.ncols()),
                got: format!("Got {} features", x.ncols()),
            });
        }

        self.assign_clusters(x, centroids)
    }

    /// Returns the learned cluster centroids.
    ///
    /// # Returns
    ///
    /// Reference to centroid matrix of shape `(k, n_features)`, or `None` if not fitted.
    pub fn centroids(&self) -> Option<&Matrix> {
        self.centroids.as_ref()
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
    fn test_kmeans_new() {
        let kmeans = KMeans::new(3, 100);
        assert_eq!(kmeans.k, 3);
        assert_eq!(kmeans.max_iter, 100);
        assert!(kmeans.centroids.is_none());
    }

    #[test]
    fn test_kmeans_empty_data() {
        let mut kmeans = KMeans::new(2, 100);
        let x = Matrix::zeros(0, 2);
        let result = kmeans.fit(&x);
        assert!(matches!(result, Err(TesseractError::EmptyTrainingData)));
    }

    #[test]
    fn test_kmeans_k_zero() {
        let mut kmeans = KMeans::new(0, 100);
        let x = matrix_from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let result = kmeans.fit(&x);
        assert!(matches!(
            result,
            Err(TesseractError::InvalidHyperparameter { .. })
        ));
    }

    #[test]
    fn test_kmeans_k_greater_than_n() {
        let mut kmeans = KMeans::new(5, 100);
        let x = matrix_from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let result = kmeans.fit(&x);
        assert!(matches!(
            result,
            Err(TesseractError::InvalidHyperparameter { .. })
        ));
    }

    #[test]
    fn test_kmeans_predict_not_fitted() {
        let kmeans = KMeans::new(2, 100);
        let x = matrix_from_vec(vec![vec![1.0, 2.0]]);
        let result = kmeans.predict(&x);
        assert!(matches!(result, Err(TesseractError::NotFitted)));
    }

    #[test]
    fn test_kmeans_simple_clustering() {
        let mut kmeans = KMeans::new(2, 100);
        // Two well-separated clusters
        let x = matrix_from_vec(vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.2, 0.2],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
            vec![10.2, 10.2],
        ]);
        kmeans.fit(&x).unwrap();

        // Predict on same data
        let assignments = kmeans.predict(&x).unwrap();
        
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
    fn test_kmeans_three_clusters() {
        let mut kmeans = KMeans::new(3, 100);
        let x = matrix_from_vec(vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![5.0, 5.0],
            vec![5.1, 5.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ]);
        kmeans.fit(&x).unwrap();

        let assignments = kmeans.predict(&x).unwrap();
        
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
    fn test_kmeans_centroids() {
        let mut kmeans = KMeans::new(2, 100);
        let x = matrix_from_vec(vec![
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![10.0, 10.0],
            vec![10.0, 10.0],
        ]);
        kmeans.fit(&x).unwrap();

        let centroids = kmeans.centroids().unwrap();
        assert_eq!(centroids.nrows(), 2);
        assert_eq!(centroids.ncols(), 2);

        // Centroids should be approximately [0, 0] and [10, 10]
        let mut found_zero = false;
        let mut found_ten = false;

        for i in 0..2 {
            let c0 = centroids[(i, 0)];
            let c1 = centroids[(i, 1)];
            
            if (c0 - 0.0).abs() < 1.0 && (c1 - 0.0).abs() < 1.0 {
                found_zero = true;
            }
            if (c0 - 10.0).abs() < 1.0 && (c1 - 10.0).abs() < 1.0 {
                found_ten = true;
            }
        }

        assert!(found_zero);
        assert!(found_ten);
    }

    #[test]
    fn test_kmeans_1d() {
        let mut kmeans = KMeans::new(2, 100);
        let x = matrix_from_vec(vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
            vec![10.0],
            vec![11.0],
            vec![12.0],
        ]);
        kmeans.fit(&x).unwrap();

        let assignments = kmeans.predict(&x).unwrap();
        
        // First 3 in one cluster, last 3 in another
        let c0 = assignments[0];
        assert_eq!(assignments[1], c0);
        assert_eq!(assignments[2], c0);
        assert_ne!(assignments[3], c0);
        assert_eq!(assignments[4], assignments[3]);
        assert_eq!(assignments[5], assignments[3]);
    }

    #[test]
    fn test_kmeans_shape_mismatch() {
        let mut kmeans = KMeans::new(2, 100);
        let x = matrix_from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        kmeans.fit(&x).unwrap();

        // Try to predict with different number of features
        let x_test = matrix_from_vec(vec![vec![1.0]]);
        let result = kmeans.predict(&x_test);
        assert!(matches!(result, Err(TesseractError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_kmeans_single_cluster() {
        let mut kmeans = KMeans::new(1, 100);
        let x = matrix_from_vec(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ]);
        kmeans.fit(&x).unwrap();

        let assignments = kmeans.predict(&x).unwrap();
        
        // All should be in cluster 0
        assert!(assignments.iter().all(|&a| a == 0));
    }

    #[test]
    fn test_kmeans_convergence() {
        let mut kmeans = KMeans::new(2, 1); // Only 1 iteration
        let x = matrix_from_vec(vec![
            vec![0.0, 0.0],
            vec![10.0, 10.0],
        ]);
        kmeans.fit(&x).unwrap();

        // Should still produce valid centroids
        assert!(kmeans.centroids().is_some());
    }

    #[test]
    fn test_kmeans_with_nan() {
        let mut kmeans = KMeans::new(2, 100);
        let x = matrix_from_vec(vec![
            vec![1.0, 2.0],
            vec![f32::NAN, 4.0],
        ]);
        let result = kmeans.fit(&x);
        assert!(matches!(result, Err(TesseractError::InvalidValue { .. })));
    }

    #[test]
    fn test_kmeans_high_dimensional() {
        let mut kmeans = KMeans::new(2, 100);
        let x = matrix_from_vec(vec![
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.1, 0.1, 0.1, 0.1, 0.1],
            vec![10.0, 10.0, 10.0, 10.0, 10.0],
            vec![10.1, 10.1, 10.1, 10.1, 10.1],
        ]);
        kmeans.fit(&x).unwrap();

        let assignments = kmeans.predict(&x).unwrap();
        
        // First 2 in one cluster, last 2 in another
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[2], assignments[3]);
        assert_ne!(assignments[0], assignments[2]);
    }

    #[test]
    fn test_kmeans_identical_points() {
        let mut kmeans = KMeans::new(2, 100);
        // All points identical
        let x = matrix_from_vec(vec![
            vec![5.0, 5.0],
            vec![5.0, 5.0],
            vec![5.0, 5.0],
            vec![5.0, 5.0],
        ]);
        kmeans.fit(&x).unwrap();

        // Should still fit without error
        let assignments = kmeans.predict(&x).unwrap();
        assert_eq!(assignments.len(), 4);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_kmeans_serialize_deserialize_json() {
        let mut kmeans = KMeans::new(2, 100);
        let x = matrix_from_vec(vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ]);
        kmeans.fit(&x).unwrap();

        // Serialize
        let serialized = serde_json::to_string(&kmeans).expect("Failed to serialize");
        assert!(!serialized.is_empty());

        // Deserialize
        let deserialized: KMeans = serde_json::from_str(&serialized).expect("Failed to deserialize");

        // Verify predictions match
        let x_test = matrix_from_vec(vec![vec![0.0, 0.0], vec![10.0, 10.0]]);
        let result_original = kmeans.predict(&x_test).unwrap();
        let result_deserialized = deserialized.predict(&x_test).unwrap();

        assert_eq!(result_original, result_deserialized);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_kmeans_serialize_unfitted() {
        let kmeans = KMeans::new(3, 50);

        // Should serialize unfitted model
        let serialized = serde_json::to_string(&kmeans).expect("Failed to serialize");
        let deserialized: KMeans = serde_json::from_str(&serialized).expect("Failed to deserialize");

        // Both should be unfitted
        let x = matrix_from_vec(vec![vec![1.0, 2.0]]);
        assert!(matches!(kmeans.predict(&x), Err(TesseractError::NotFitted)));
        assert!(matches!(deserialized.predict(&x), Err(TesseractError::NotFitted)));
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_kmeans_roundtrip_preserves_state() {
        let mut kmeans = KMeans::new(2, 100);
        let x = matrix_from_vec(vec![
            vec![1.0, 2.0],
            vec![1.1, 2.1],
            vec![9.0, 8.0],
            vec![9.1, 8.1],
        ]);
        kmeans.fit(&x).unwrap();

        // Serialize and deserialize
        let serialized = serde_json::to_string(&kmeans).unwrap();
        let deserialized: KMeans = serde_json::from_str(&serialized).unwrap();

        // State should be preserved
        assert_eq!(kmeans.k, deserialized.k);
        assert_eq!(kmeans.max_iter, deserialized.max_iter);
        
        // Centroids should match
        let c1 = kmeans.centroids().unwrap();
        let c2 = deserialized.centroids().unwrap();
        assert_eq!(c1.nrows(), c2.nrows());
        assert_eq!(c1.ncols(), c2.ncols());
    }

    #[test]
    fn test_kmeans_k_getter() {
        let kmeans = KMeans::new(5, 100);
        assert_eq!(kmeans.k(), 5);
    }
}
