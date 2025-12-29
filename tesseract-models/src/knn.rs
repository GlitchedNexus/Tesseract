use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use tesseract_core::{Matrix, Predictions, Result, TesseractError};

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
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fit(&mut self, x: &Matrix, y: &Predictions, k: usize) {
        self.dataset = Some(x.clone());
        self.labels = Some(y.clone());
        self.k = k;
    }

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
                // squared euclidean distance without allocating aj - xi
                let mut dist2: f32 = 0.0;
                for (&av, &xv) in aj.iter().zip(xi.iter()) {
                    let diff = av - xv;
                    dist2 += diff * diff;
                }

                // If NaNs are possible, you can skip or error:
                if dist2.is_nan() {
                    continue; // or: return Err(TesseractError::InvalidValue { ... })
                }

                let cand = Neighbour { dist2, idx: j };

                if heap.len() < self.k {
                    heap.push(cand);
                } else if let Some(&worst) = heap.peek() {
                    // worst is the largest dist2 among kept neighbors
                    if cand.dist2 < worst.dist2 {
                        heap.pop();
                        heap.push(cand);
                    }
                }
            }

            // Majority vote over heap items (k neighbors)
            counts.clear();
            let mut best_label: usize = 0;
            let mut best_count: usize = 0;
            let mut best_closest: f32 = f32::INFINITY; // tie-break by closeness

            for n in heap.into_iter() {
                let label = y[n.idx];
                let c = counts.entry(label).or_insert(0);
                *c += 1;

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
