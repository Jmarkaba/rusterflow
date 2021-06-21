use rand::distributions::Standard;

use ndarray::{Array, Array2};
use ndarray::linalg::general_mat_mul;
use ndarray_rand::RandomExt;

#[derive(Debug)]
struct Network {
    num_layers: usize,
    sizes: Vec<usize>,
    biases: Vec<Array2<f64>>,
    weights: Vec<Array2<f64>>,
}

enum Activation {Crossentropy}

impl Network {
       fn new(sizes: &[usize]) -> Network {

        let num_layers = sizes.len();
        let mut biases: Vec<Array2<f64>> = Vec::new();
        let mut weights: Vec<Array2<f64>> = Vec::new();

        for i in 1..num_layers {
            biases.push(Array::random((sizes[i], 1), Standard));
            weights.push(Array::random((sizes[i], sizes[i - 1]), Standard));
        }

        Network {
            num_layers: num_layers,
            sizes: sizes.to_owned(),
            biases: biases,
            weights: weights,
        }
    }

    fn calc_z(&self, layer: usize, input: &Array2<f64>) -> Array2<f64> {
        let mut out: Array2<f64> = self.biases[layer].clone();
        let mut weight = self.weights[layer].clone();
        weight.swap_axes(0, 1);
        general_mat_mul(1.0, &weight, input, 1.0, &mut out);
        return out;
    }

        
    fn sigmoid(z: f64) -> f64 {
        return 1.0 / (1.0 + (-z).exp());
    }

    fn predict(&self, input: Array2<f64>) {
        for i in 1..self.sizes.len() {

        }
    }

    fn update(/*batch, learning rate*/) {


    }
}

fn main() {

}