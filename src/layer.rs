mod activation;

use ndarray::Array2;

type Shape = Array2<usize>;

trait Layer {
    fn new<T>(&self, in_size: &Shape, out_size: &Shape, meta: Option<T>) -> Self;
    fn sizes() -> Shape;
    fn forward(&self, input: &Array2<f64>) -> Array2<f64>;
    fn backward(&self, input: &Array2<f64>) -> Array2<f64>;
}

//Dense Layer
struct DenseLayer<A: ActivationDFJKDF> {
    in_size: usize,
    out_size: usize,
    weights: Array2<f64>,
    partials: Array2<f64>,
    activation: A,
}

impl Layer for DenseLayer {

}
