//type MatrixVector = Array2<f64>; Just a thought to make code more readable, not necessary at all
use ndarray::Array2;

pub trait LossTrait {
    fn loss(&self, predicted: &Array2<f64>, actual: &Array2<f64>) -> f64;
    fn loss_gradient(&self, predicted: &Array2<f64>, actual: &Array2<f64>) -> Array2<f64>;
    fn new() -> Self where Self: Sized;
}

//Cross Entropy
pub struct CrossEntropy;

impl LossTrait for CrossEntropy {
    fn loss(&self, predicted: &Array2<f64>, actual: &Array2<f64>) -> f64 {
        let mut sum: f64 = 0.0;

        for i in 1..predicted.len() {
            sum -= actual[[i, 0]] * predicted[[i, 0]].log2(); // Review cross-entropy loss.
        }

        sum
    }

    fn loss_gradient(&self, predicted: &Array2<f64>, actual: &Array2<f64>) -> Array2<f64> {
        //Fix
        return predicted.clone();
    }

    fn new() -> Self where Self: Sized {
        CrossEntropy
    }
}

//Square
pub struct Square;

impl LossTrait for Square {
    fn loss(&self, predicted: &Array2<f64>, actual: &Array2<f64>) -> f64 {
        let mut sum: f64 = 0.0;

        for i in 1..predicted.len() {
            sum += (predicted[[i, 0]] - actual[[i, 0]]).powi(2);
        }

        sum
    }

    fn loss_gradient(&self, predicted: &Array2<f64>, actual: &Array2<f64>) -> Array2<f64> {
        2.0 * (actual - predicted)
    }

    fn new() -> Self where Self: Sized {
        Square
    }
}
