use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use std::hint::black_box;
use tesseract_core::{Matrix, Vector};
use tesseract_models::decision_stump::DecisionStump;
use tesseract_models::decision_tree::DecisionTree;
use tesseract_models::k_means::KMeans;
use tesseract_models::k_medians::KMedians;
use tesseract_models::knn::KNN;
use tesseract_models::linear_regression::LinearRegression;

fn make_regression_data(rows: usize, cols: usize) -> (Matrix, Vector) {
    let x = Matrix::from_fn(rows, cols, |i, j| ((i * (j + 1)) % 97) as f32 * 0.01);
    let y = Vector::from_iterator(
        rows,
        (0..rows).map(|i| {
            let i_f = i as f32;
            0.5 * i_f + ((i % 11) as f32) * 0.25 + 3.0
        }),
    );
    (x, y)
}

fn make_classification_data(rows: usize, cols: usize, classes: usize) -> (Matrix, Vec<usize>) {
    let x = Matrix::from_fn(rows, cols, |i, j| {
        let base = ((i + 3 * j) % 101) as f32 * 0.01;
        let class_offset = (i % classes) as f32 * 0.2;
        base + class_offset
    });
    let y = (0..rows).map(|i| i % classes).collect();
    (x, y)
}

fn make_cluster_data(points_per_cluster: usize, dims: usize, clusters: usize) -> Matrix {
    let rows = points_per_cluster * clusters;
    Matrix::from_fn(rows, dims, |i, j| {
        let cluster = i / points_per_cluster;
        let within = i % points_per_cluster;
        let center = cluster as f32 * 8.0;
        center + (within as f32 * 0.01) + (j as f32 * 0.05)
    })
}

fn bench_linear_regression(c: &mut Criterion) {
    let (x, y) = make_regression_data(512, 8);

    let mut group = c.benchmark_group("linear_regression");
    group.bench_function("fit_512x8", |b| {
        b.iter_batched(
            || (x.clone(), y.clone(), LinearRegression::new()),
            |(x_local, y_local, mut model)| {
                model.fit(black_box(&x_local), black_box(&y_local)).unwrap();
            },
            BatchSize::SmallInput,
        )
    });

    let mut fitted = LinearRegression::new();
    fitted.fit(&x, &y).unwrap();
    group.bench_function("predict_512x8", |b| {
        b.iter(|| {
            let preds = fitted.predict(black_box(&x)).unwrap();
            black_box(preds);
        })
    });
    group.finish();
}

fn bench_knn(c: &mut Criterion) {
    let (x_train, y_train) = make_classification_data(2_000, 12, 3);
    let (x_query, _) = make_classification_data(256, 12, 3);

    let mut model = KNN::new();
    model.fit(&x_train, &y_train, 5);

    c.bench_function("knn_predict_k5_2000x12_query256", |b| {
        b.iter(|| {
            let preds = model.predict(black_box(&x_query)).unwrap();
            black_box(preds);
        })
    });
}

fn bench_decision_stump(c: &mut Criterion) {
    let (x, y) = make_classification_data(2_048, 10, 3);

    let mut group = c.benchmark_group("decision_stump");
    group.bench_function("fit_2048x10", |b| {
        b.iter_batched(
            || (x.clone(), y.clone(), DecisionStump::new()),
            |(x_local, y_local, mut model)| {
                model.fit(black_box(&x_local), black_box(&y_local)).unwrap();
            },
            BatchSize::SmallInput,
        )
    });

    let mut fitted = DecisionStump::new();
    fitted.fit(&x, &y).unwrap();
    group.bench_function("predict_2048x10", |b| {
        b.iter(|| {
            let preds = fitted.predict(black_box(&x)).unwrap();
            black_box(preds);
        })
    });
    group.finish();
}

fn bench_decision_tree(c: &mut Criterion) {
    let (x, y) = make_classification_data(1_024, 10, 3);

    let mut group = c.benchmark_group("decision_tree");
    group.bench_function("fit_1024x10_depth8", |b| {
        b.iter_batched(
            || (x.clone(), y.clone(), DecisionTree::new(8, 2, 1)),
            |(x_local, y_local, mut model)| {
                model.fit(black_box(&x_local), black_box(&y_local)).unwrap();
            },
            BatchSize::SmallInput,
        )
    });

    let mut fitted = DecisionTree::new(8, 2, 1);
    fitted.fit(&x, &y).unwrap();
    group.bench_function("predict_1024x10_depth8", |b| {
        b.iter(|| {
            let preds = fitted.predict(black_box(&x)).unwrap();
            black_box(preds);
        })
    });
    group.finish();
}

fn bench_kmeans(c: &mut Criterion) {
    let x = make_cluster_data(300, 6, 4);

    let mut group = c.benchmark_group("kmeans");
    group.bench_function("fit_1200x6_k4", |b| {
        b.iter_batched(
            || (x.clone(), KMeans::new(4, 50)),
            |(x_local, mut model)| {
                model.fit(black_box(&x_local)).unwrap();
            },
            BatchSize::SmallInput,
        )
    });

    let mut fitted = KMeans::new(4, 50);
    fitted.fit(&x).unwrap();
    group.bench_function("predict_1200x6_k4", |b| {
        b.iter(|| {
            let preds = fitted.predict(black_box(&x)).unwrap();
            black_box(preds);
        })
    });
    group.finish();
}

fn bench_kmedians(c: &mut Criterion) {
    let x = make_cluster_data(300, 6, 4);

    let mut group = c.benchmark_group("kmedians");
    group.bench_function("fit_1200x6_k4", |b| {
        b.iter_batched(
            || (x.clone(), KMedians::new(4, 50)),
            |(x_local, mut model)| {
                model.fit(black_box(&x_local)).unwrap();
            },
            BatchSize::SmallInput,
        )
    });

    let mut fitted = KMedians::new(4, 50);
    fitted.fit(&x).unwrap();
    group.bench_function("predict_1200x6_k4", |b| {
        b.iter(|| {
            let preds = fitted.predict(black_box(&x)).unwrap();
            black_box(preds);
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_linear_regression,
    bench_knn,
    bench_decision_stump,
    bench_decision_tree,
    bench_kmeans,
    bench_kmedians
);
criterion_main!(benches);
