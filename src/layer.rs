use ndarray::{Array, Array2, Axis, concatenate};
use rand::distributions::Standard;
use ndarray_rand::RandomExt;
use crate::activation::*;
use ndarray::s;

type Shape = Array2<usize>;
// type MVector = Array2<f64>;
// type MMatrix = Array2<f64>;

pub trait Layer {
    //fn new<T>(&self, in_size: Shape, out_size: Shape, meta: Option<T>) -> Self;
    //fn sizes() -> Shape;
    fn forward(&mut self, input: &Array2<f64>) -> Array2<f64>;
    fn backward(&mut self, input: &Array2<f64>) -> Array2<f64>;
    fn update(&mut self, learning_rate: f64);
}

//Dense Layer
pub struct DenseLayer<A: Activation> {
    in_size: usize,
    out_size: usize,
    weights: Array2<f64>,
    partials: Array2<f64>,
    X: Array2<f64>,
    Z: Array2<f64>,
    activation: A,
}

impl<A: Activation> DenseLayer<A> {
    pub fn new(in_size: Shape, out_size: Shape, activation: A) -> Self {
        let in_size = in_size[[0, 0]];
        let out_size = out_size[[0, 0]];
        let mut weights = Array::random((in_size + 1, out_size), Standard);
        // let mut partials = Array::zeros((in_size + 1, out_size));
        // let mut x = Array::zeros((1, in_size + 1));
        // let mut z = Array::zeros((1, out_size));
        let mut partials = Array::random((in_size + 1, out_size), Standard);
        let mut x = Array::random((1, in_size + 1), Standard);
        let mut z = Array::random((1, out_size), Standard);


        //DEBUG
        //println!("INSIZE: {}\n OUTSIZE: {}\n", in_size, out_size);


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
        let extended_input = concatenate(Axis(1), &[input.view(), one.view()]);

        self.Z = match extended_input {
            Ok(inp) => {
                self.X = inp;
                self.X.dot(&self.weights)
            }
            Err(_) => panic!("Unable to concatenate input"),
        };

        self.activation.vectorized(&self.Z)
    }

    fn backward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        //let dJ_dz = self.activation.gradient(&self.Z) * input;
        let a = self.activation.gradient(&self.Z);

        //DEBUG
        //println!("a.size(): {}, {} / input.size(): {}, {}\n", a.dim().0, a.dim().1, input.dim().0, input.dim().1);

        let dJ_dz = a * input;
        self.partials = self.X.t().dot(&dJ_dz);
        let weights_slice = &self.weights.slice(s![0..self.in_size, ..]);
        dJ_dz.dot(&weights_slice.t()) //dJ/dx
    }

    fn update(&mut self, learning_rate: f64) {
        self.weights = &self.weights - &(&self.partials * learning_rate);
    }
}
