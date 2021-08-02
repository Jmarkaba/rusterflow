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

        for i in 0..predicted.len() {
            sum -= actual[[0, i]] * predicted[[0, i]].log2(); // Review cross-entropy loss.
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

        for i in 0..predicted.len() {
            sum += (predicted[[0, i]] - actual[[0, i]]).powi(2);

            //DEBUG
            //println!("i: {}, PREDICTED: {}, ACTUAL: {}", i, predicted[[0, i]], actual[[0, i]]);
        }

        sum
    }

    fn gradient(&self, predicted: &Array2<f64>, actual: &Array2<f64>) -> Array2<f64> {
        2.0 * (predicted - actual)
    }
}
