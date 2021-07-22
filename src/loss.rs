use ndarray::Array2;

pub trait Loss {
    fn new() -> Self where Self: Sized;
    fn value(&self, predicted: &Array2<f64>, actual: &Array2<f64>) -> f64;
    fn gradient(&self, predicted: &Array2<f64>, actual: &Array2<f64>) -> Array2<f64>;
}

//Cross Entropy
pub struct CrossEntropy;

impl Loss for CrossEntropy {
    fn new() -> Self where Self: Sized {
        CrossEntropy
    }

    fn value(&self, predicted: &Array2<f64>, actual: &Array2<f64>) -> f64 {
        let mut sum: f64 = 0.0;

        for i in 1..predicted.len() {
            sum -= actual[[i, 0]] * predicted[[i, 0]].log2(); // Review cross-entropy loss.
        }

        sum
    }

    fn gradient(&self, predicted: &Array2<f64>, actual: &Array2<f64>) -> Array2<f64> {
        //Fix
        return predicted.clone();
    }
}

//Square
pub struct Square;

impl Loss for Square {
    fn new() -> Self where Self: Sized {
        Square
    }

    fn value(&self, predicted: &Array2<f64>, actual: &Array2<f64>) -> f64 {
        let mut sum: f64 = 0.0;

        for i in 1..predicted.len() {
            sum += (predicted[[i, 0]] - actual[[i, 0]]).powi(2);
        }

        sum
    }

    fn gradient(&self, predicted: &Array2<f64>, actual: &Array2<f64>) -> Array2<f64> {
        2.0 * (actual - predicted)
    }
}
