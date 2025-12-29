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
