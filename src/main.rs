use rand::distributions::Standard;

use ndarray::{Array, Array2};
use ndarray::linalg::general_mat_mul;
use ndarray_rand::RandomExt;

//use nalgebra::matrix;

use activation::ActivationTrait;
use loss::LossTrait;

//#[derive(Debug)]
struct Network<'a, 'b> {
    num_layers: usize,
    sizes: Vec<usize>,
    biases: Vec<Array2<f64>>,
    weights: Vec<Array2<f64>>,
    activation: &'a dyn ActivationTrait, //dyn means dynamic trait in this case
    loss: &'b dyn LossTrait, //dyn is not needed, but rust gives warnings to make it more cleaner/explicit code
}

impl<'a, 'b> Network<'a, 'b> { //I have no idea why adding the <'a, 'b>s makes this code compile lmao
       fn new(sizes: &[usize]) -> Network {

        let num_layers = sizes.len();
        let mut biases: Vec<Array2<f64>> = Vec::new();
        let mut weights: Vec<Array2<f64>> = Vec::new();

        for i in 1..num_layers {
            biases.push(Array::random((sizes[i], 1), Standard));
            weights.push(Array::random((sizes[i], sizes[i - 1]), Standard));
        }

        //Assume activation is relu and loss is crossentropy
        Network {
            num_layers: num_layers,
            sizes: sizes.to_owned(),
            biases: biases,
            weights: weights,
            activation: &activation::Sigmoid,
            loss: &loss::CrossEntropy,
        }
    }

    fn predict() {

    }

    fn update() {

    }
}

//OLD CODE FROM NETWORK STRUCT
    // fn calc_z(&self, layer: usize, input: &Array2<f64>) -> Array2<f64> {
    //     let mut out: Array2<f64> = self.biases[layer].clone();
    //     let mut weight = self.weights[layer].clone();
    //     weight.swap_axes(0, 1);
    //     general_mat_mul(1.0, &weight, input, 1.0, &mut out);
    //     return out;
    // }
    //
    // fn sigmoid(&self, z: f64) -> f64 {
    //     return 1.0 / (1.0 + (-z).exp());
    // }
    //
    // fn relu(&self, z: f64) -> f64 {
    //     if (z < 0.0) {
    //         0.0
    //     } else {
    //         z
    //     }
    // }
    //
    // fn relu_gradient(&self, z: f64) -> f64 {
    //     if (z < 0.0) {
    //         0.0
    //     } else {
    //         1.0
    //     }
    // }

    // fn calc_z(&self, layer: usize, input: &Array2<f64>) -> Array2<f64> {
    //     let mut out: Array2<f64> = self.biases[layer].clone();
    //     let mut weight = self.weights[layer].clone();
    //     weight.swap_axes(0, 1);
    //     general_mat_mul(1.0, &weight, input, 1.0, &mut out);
    //     return out;
    // }
    //
    // fn sigmoid(&self, z: f64) -> f64 {
    //     return 1.0 / (1.0 + (-z).exp());
    // }
    //
    // fn relu(&self, z: f64) -> f64 {
    //     if (z < 0.0) {
    //         0.0
    //     } else {
    //         z
    //     }
    // }
    //
    // fn relu_gradient(&self, z: f64) -> f64 {
    //     if (z < 0.0) {
    //         0.0
    //     } else {
    //         1.0
    //     }
    // }
    //
    // fn cross_entropy(&self, prediction: &Array2<f64>, sample: Array2<f64>) -> f64 {
    //     1.0
    // }
    //
    // fn cross_entropy_loss(&self, prediction: &Array2<f64>, batch: Vec<Array2<f64>>) -> f64{
    //     // Running sum.
    //     let mut sum: f64 = 0.0;
    //
    //     // Iterate over the training samples in the batch.
    //     for sample in batch {
    //         // Add loss of training sample to running sum.
    //         sum += self.cross_entropy(prediction, sample);
    //     }
    //
    //     sum
    // }
    //
    // fn square(&self, prediction: &Array2<f64>, sample: Array2<f64>) -> f64 {
    //     for i in 1..prediction.len()
    //     ;
    // }

    //TODO: UPDATE PREDCTION FUNCTION

    //Return prediction vector
    //Uses sigmoid activation function
    // fn predict(&self, mut input: Array2<f64>) -> Array2<f64> {
    //     //Iterates through every layer and calculates z
    //     for i in 1..self.sizes.len() {
    //         //Calculate z for layer i
    //         let z: Array2<f64> = self.calc_z(i, &input);
    //
    //         //Apply activation function element-wise.
    //         for j in 1..z.len() {
    //             input[[j, 0]] = self.sigmoid(z[[j, 0]]);
    //         }
    //     }
    //
    //     input
    // }
    //
    // fn update(/*batch, learning rate*/) {
    //     //
    // }

//Activation Functions
pub mod activation {
    use ndarray::Array2;

    pub trait ActivationTrait {
        fn activation(&self, z: f64) -> f64;
        fn activation_gradient(&self, z: f64) -> f64;
        fn activation_vectorized(&self, z: &mut Array2<f64>);
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
    }

    //ReLU
    pub struct Relu;

    impl ActivationTrait for Relu {
        fn activation(&self, z: f64) -> f64 {
            if (z < 0.0) {
                0.0
            } else {
                z
            }
        }

        fn activation_gradient(&self, z: f64) -> f64 {
            if (z < 0.0) {
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
    }
}

//Loss Functions
pub mod loss {
    //type MatrixVector = Array2<f64>; Just a thought to make code more readable, not necessary at all
    use ndarray::Array2;

    pub trait LossTrait {
        fn loss(&self, predicted: Array2<f64>, actual: Array2<f64>) -> f64;
    }

    //Cross Entropy
    pub struct CrossEntropy;

    impl LossTrait for CrossEntropy {
        fn loss(&self, predicted: Array2<f64>, actual: Array2<f64>) -> f64 {
            let mut sum: f64 = 0.0;

            for i in 1..predicted.len() {
                sum -= actual[[i, 0]] * predicted[[i, 0]].log2(); // Review cross-entropy loss.
            }

            sum
        }
    }

    //Square
    pub struct Square;

    impl LossTrait for Square {
        fn loss(&self, predicted: Array2<f64>, actual: Array2<f64>) -> f64 {
            let mut sum: f64 = 0.0;

            for i in 1..predicted.len() {
                sum += (predicted[[i, 0]] - actual[[i, 0]]).powi(2);
            }

            sum
        }
    }
}

fn main() {

}
