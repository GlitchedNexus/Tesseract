use std::collections::{HashMap};

use crate::{Matrix, Predictions, Result, TesseractError};

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
        match (&self.dataset, &self.labels) {
            (Some(a), Some(b)) => {
                let n = x.nrows();
                let d = x.ncols();

                if a.ncols() != d {
                    return Err(crate::TesseractError::ShapeMismatch {
                        expected: format!("Expected {} features", a.ncols()),
                        got: format!("Got {} features", d),
                    });
                }

                let mut predictions: Predictions = vec![0usize; n];

                for (i, prediction) in predictions.iter_mut().enumerate() {
                    let xi = x.row(i);

                    let mut distances: HashMap<usize, f32> = HashMap::new();

                    for (j, aj) in a.row_iter().enumerate() {
                        let euclidian_distance = (aj - xi).norm_squared();
                        distances.insert(j, euclidian_distance);
                    }

                    if distances.len() < self.k {
                        return Err(TesseractError::InsufficientTrainingData);
                    }

                    let neighbours = k_smallest(&distances, self.k);

                    let mut counts: HashMap<usize, usize> = HashMap::new();
                    let mut is_first_class_set = false;
                    let mut pred: usize = 0;
                    let mut max_count = 0;

                    for (index, _) in neighbours {
                        let label = b[index];

                        let count = counts.get(&label);

                        match count {
                            Some(&num) => {
                                if num + 1 > max_count {
                                    pred = label;
                                    max_count = num + 1;
                                    counts.insert(pred, max_count);
                                }

                                counts.insert(label, num + 1);
                            }

                            _ => {
                                counts.insert(label, 1);

                                match is_first_class_set {
                                    false => {
                                        pred = label;
                                        max_count += 1;
                                        is_first_class_set = true;
                                    }
                                    true => {}
                                }
                            }
                        }
                    }

                    *prediction = pred;
                }

                Ok(predictions)
            }
            (_, _) => return Err(crate::TesseractError::NotFitted),
        }
    }
}

fn k_smallest(map: &HashMap<usize, f32>, k: usize) -> Vec<(usize, f32)> {
    let mut v: Vec<(usize, f32)> = map.iter().map(|(&i, &d)| (i, d)).collect();

    v.select_nth_unstable_by(k, |a, b| a.1.partial_cmp(&b.1).unwrap());

    v.truncate(k);

    v
}
