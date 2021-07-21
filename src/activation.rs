use ndarray::Array2;

pub trait ActivationFn {
    fn activation(&self, z: f64) -> f64;
    fn activation_partial(&self, z: f64) -> f64;
    fn activation_partial_vectorized(&self, z: &mut Array2<f64>);
    fn new() -> Self where Self: Sized;
}

//Sigmoid
pub struct Sigmoid;

impl ActivationTrait for Sigmoid {
    fn activation(&self, z: f64) -> f64 {
        return 1.0 / (1.0 + (-z).exp());
    }

    fn activation_gradient(&self, z: f64) -> f64 {
        self.activation(z) * (1.0 - self.activation(z))
    }

    fn activation_vectorized(&self, z: &mut Array2<f64>) {
        for i in 1..z.len() {
            z[[i, 0]] = self.activation(z[[i, 0]]);
        }
    }

    fn new() -> Self where Self: Sized {
        Sigmoid
    }
}

//ReLU
pub struct Relu;

impl ActivationTrait for Relu {
    fn activation(&self, z: f64) -> f64 {
        if z < 0.0 {
            0.0
        } else {
            z
        }
    }

    fn activation_gradient(&self, z: f64) -> f64 {
        if z < 0.0 {
            0.0
        } else {
            1.0
        }
    }

    fn activation_vectorized(&self, z: &mut Array2<f64>) {
        for i in 1..z.len() {
            z[[i, 0]] = self.activation(z[[i, 0]]);
        }
    }

    fn new() -> Self where Self: Sized {
        Relu
    }
}
