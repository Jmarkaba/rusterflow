use ndarray::{Array, Array2, Axis, concatenate};
use rand::distributions::Standard;
use ndarray_rand::RandomExt;
use crate::activation::*;

type Shape = Array2<usize>;

pub trait Layer {
    //fn new<T>(&self, in_size: Shape, out_size: Shape, meta: Option<T>) -> Self;
    //fn sizes() -> Shape;
    fn forward(&mut self, input: &Array2<f64>) -> Array2<f64>;
    fn backward(&mut self, input: &Array2<f64>) -> Array2<f64>;
}

//Dense Layer
struct DenseLayer<A: Activation> {
    in_size: usize,
    out_size: usize,
    weights: Array2<f64>,
    partials: Array2<f64>,
    X: Array2<f64>,
    Z: Array2<f64>,
    activation: A,
}

impl<A: Activation> DenseLayer<A> {
    fn new(&self, in_size: Shape, out_size: Shape, activation: A) -> Self {
        let in_size = in_size[[0, 0]];
        let out_size = out_size[[0, 0]];
        let mut weights = Array::random((in_size + 1, out_size), Standard);
        let mut partials = Array::zeros((in_size + 1, out_size));
        let mut x = Array::zeros((1, out_size));
        let mut z = Array::zeros((1, out_size));

        Self {
            in_size: in_size,
            out_size: out_size,
            weights: weights,
            partials: partials,
            X: x,
            Z: z,
            activation: activation,
        }
    }
}

impl<A: Activation> Layer for DenseLayer<A> {
    // fn sizes() -> Shape {
    //
    // }

    fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        let one = Array::ones((1, 1));
        let extended_input = concatenate(Axis(0), &[input.view(), one.view()]);

        self.Z = match extended_input {
            Ok(inp) => inp.dot(&self.weights),
            Err(_) => panic!("Unable to concatenate input"),
        };

        self.activation.vectorized(&self.Z)
    }

    fn backward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        let dJ_dz = self.activation.gradient(&self.Z) * input;
        self.partials = &dJ_dz * &self.X;
        dJ_dz * &self.weights //dJ/dx
    }
}
