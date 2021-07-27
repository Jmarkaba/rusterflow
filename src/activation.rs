use ndarray::Array2;

pub trait Activation {
    fn new() -> Self where Self: Sized;
    fn value(&self, z: f64) -> f64;
    fn partial(&self, z: f64) -> f64;
    fn vectorized(&self, z: &Array2<f64>) -> Array2<f64>;
    fn gradient(&self, z: &Array2<f64>) -> Array2<f64>;
}

//Sigmoid
pub struct Sigmoid;

impl Activation for Sigmoid {
    fn new() -> Self where Self: Sized {
        Sigmoid
    }

    fn value(&self, z: f64) -> f64 {
        return 1.0 / (1.0 + (-z).exp());
    }

    fn partial(&self, z: f64) -> f64 {
        self.value(z) * (1.0 - self.value(z))
    }

    fn vectorized(&self, z: &Array2<f64>) -> Array2<f64> {
        let mut a = z.clone();

        for i in 1..a.len() {
            a[[i, 0]] = self.value(a[[i, 0]]);
        }

        a
    }

    fn gradient(&self, z: &Array2<f64>) -> Array2<f64> {
        let mut a = z.clone();

        for i in 1..a.len() {
            a[[i, 0]] = self.partial(a[[i, 0]]);
        }

        a
    }
}

//ReLU
pub struct Relu;

impl Activation for Relu {
    fn new() -> Self where Self: Sized {
        Relu
    }

    fn value(&self, z: f64) -> f64 {
        if z < 0.0 {
            0.0
        } else {
            z
        }
    }

    fn partial(&self, z: f64) -> f64 {
        if z < 0.0 {
            0.0
        } else {
            1.0
        }
    }

    fn vectorized(&self, z: &Array2<f64>) -> Array2<f64> {
        let mut a = z.clone();

        for i in 1..a.len() {
            a[[i, 0]] = self.value(a[[i, 0]]);
        }

        a
    }

    fn gradient(&self, z: &Array2<f64>) -> Array2<f64> {
        let mut a = z.clone();

        for i in 1..a.len() {
            a[[i, 0]] = self.partial(a[[i, 0]]);
        }

        a
    }
}
