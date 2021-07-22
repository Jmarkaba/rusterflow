use ndarray::Array2;
mod loss;

use ndarray::Array2;
use layer::*;
use loss::*;

struct Network<L: Loss, A: Layer> {
    num_layers: usize,
    layers: Vec<A>,
    loss: L,
}

impl<L: Loss, A: Layer> Network<L, A> {
    //TODO:
    fn new(sizes: &[usize]) -> Self {
        //
        // let num_layers = sizes.len();
        // let mut biases: Vec<Array2<f64>> = Vec::new();
        // let mut weights: Vec<Array2<f64>> = Vec::new();
        //
        // for i in 1..num_layers {
        //     biases.push(Array::random((sizes[i], 1), Standard));
        //     weights.push(Array::random((sizes[i], sizes[i - 1]), Standard));
        // }
        //
        Self {
            num_layers: 420,
            layers: Vec::new(),
            loss: L::new()
        }
    }

    //TODO
    fn calc_z(&self, input: &Array2<f64>, layer: usize) -> Array2<f64> {
        // W[layer] * X + B[layer]
        // self.weights[layer].dot(input) + &self.biases[layer]
        input.clone()
    }

    fn batch_gradient(&self, batch: &Vec<(Array2<f64>, Array2<f64>)>) -> Array2<f64> {
        let n = batch.len();
        // let sum_vector = ArrayBase::zeros(batch[0].1.shape());
        for datum in batch {
            let prediction = self.predict(&datum.0);
            let loss_gradient = self.loss.gradient(&prediction, &datum.1);


        }

        return batch[0].0.clone(); //Fix
    }

    //TODO
    fn predict(&self, input: &Array2<f64>) -> Array2<f64> {
        // Let the initial X be the input.
        let mut x = input.clone();

        // Iterate over the layers:
        for i in 1..self.num_layers {
            // X[i] = W[i-1] * X[i-1] + B[i-1]
            x = self.calc_z(&x, i - 1);

            //Apply activation function over each node in the layer
            //self.activation.activation_vectorized(&mut x);
        }

        //Return prediction
        x
    }

    fn update(&self, batch: &Vec<(Array2<f64>, Array2<f64>)>, learning_rate: f64, epochs: usize) {
        // The batch cost function will be the average of the costs evaluated at each
        // of the training examples.

        // Take the gradient of the batch cost function with respect to the output vector
        // by averaging the gradients of the losses of each training example in the batch.

        // Iterate over the layers and do the following:
            // Multiply the gradient with respect to the activations
            // by the gradient of the activations with respect to z.

            //
    }
}
