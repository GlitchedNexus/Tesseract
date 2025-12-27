use tesseract::models::decision_stump::DecisionStump;
use tesseract::{Matrix, Predictions};

fn main() {
    let data = Matrix::zeros(4, 4);
    let y: Predictions = vec![0, 0, 0, 0];
    let mut stump = DecisionStump::new();

    stump.fit(&data, &y);
}
