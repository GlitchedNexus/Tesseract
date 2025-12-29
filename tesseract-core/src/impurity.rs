use crate::Float;

/// Computes the **Gini impurity** of a node given class counts.
///
/// Gini impurity measures how often a randomly chosen element from the node
/// would be misclassified if it were labeled according to the class
/// distribution in that node.
///
/// # Definition
///
/// For class counts `c_k` with total samples `n`, Gini impurity is defined as:
///
/// ```text
/// G = 1 − Σ_k (c_k / n)²
/// ```
///
/// - `G = 0` indicates a **pure** node (all samples belong to one class)
/// - Larger values indicate greater class mixing
///
/// This function is typically used when evaluating **decision stumps**,
/// **decision trees**, and other **CART-style models**.
///
/// # Parameters
///
/// - `counts`: A slice of class counts, where `counts[k]` is the number of
///   samples belonging to class `k`.
/// - `n`: The total number of samples in the node. This should equal
///   `counts.iter().sum()`.
///
/// # Returns
///
/// - The Gini impurity as a floating-point value in `[0, 1]`.
/// - Returns `0.0` if `n == 0` (empty node).
///
/// # Panics
///
/// This function does **not** panic, but passing an `n` value that does not
/// match the sum of `counts` will result in an incorrect impurity value.
///
/// # Examples
///
/// ```rust
/// use tesseract_core::impurity::gini_from_counts;
///
/// let counts = vec![3, 1]; // 3 samples of class 0, 1 sample of class 1
/// let gini = gini_from_counts(&counts, 4);
///
/// assert!((gini - 0.375).abs() < 1e-6);
/// ```
///
/// # Notes
///
/// - This implementation avoids logarithms and is faster than entropy-based
///   impurity measures.
/// - It is the default split criterion used in **CART** (Classification And
///   Regression Trees).
pub fn gini_from_counts(counts: &[usize], n: usize) -> Float {
    if n == 0 {
        return 0.0;
    }

    let n_f = n as Float;
    let mut sum_sq = 0.0;

    for &c in counts {
        let p = c as Float / n_f;
        sum_sq += p * p;
    }

    1.0 - sum_sq
}
