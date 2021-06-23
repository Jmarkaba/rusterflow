use rand::distributions::Standard;

use ndarray::{Array, Array2};
use ndarray::linalg::general_mat_mul;
use ndarray_rand::RandomExt;

use nalgebra::matrix;

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

    fn sigmoid(&self, z: f64) -> f64 {
        return 1.0 / (1.0 + (-z).exp());
    }

    fn relu(&self, z: f64) -> f64 {
        if (z < 0.0) {
            0.0
        } else {
            z
        }
    }

    fn relu_gradient(&self, z: f64) -> f64 {
        if (z < 0.0) {
            0.0
        } else {
            1.0
        }
    }

    fn cross_entropy(&self, prediction: &Array2<f64>, sample: Array2<f64>) -> f64 {
        1.0
    }

    fn cross_entropy_loss(&self, prediction: &Array2<f64>, batch: Vec<Array2<f64>>) -> f64{
        // Running sum.
        let mut sum: f64 = 0.0;

        // Iterate over the training samples in the batch.
        for sample in batch {
            // Add loss of training sample to running sum.
            sum += self.cross_entropy(prediction, sample);
        }

        sum
    }

    //Return prediction vector
    //Uses sigmoid activation function
    fn predict(&self, mut input: Array2<f64>) -> Array2<f64> {
        //Iterates through every layer and calculates z
        for i in 1..self.sizes.len() {
            //Calculate z for layer i
            let z: Array2<f64> = self.calc_z(i, &input);

            //Apply activation function element-wise.
            for j in 1..z.len() {
                input[[j, 0]] = self.sigmoid(z[[j, 0]]);
            }
        }

        input
    }

    fn update(/*batch, learning rate*/) {
        //
    }
}

fn main() {

}
