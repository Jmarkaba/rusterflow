use rand::distributions::Standard;

use ndarray::{Array, Array2};
use ndarray::linalg::general_mat_mul;
use ndarray_rand::RandomExt;
use ndarray::arr2;

//use nalgebra::matrix;

use activation::*;
use loss::*;

//#[derive(Debug)]
struct Network<A: ActivationTrait, L: LossTrait> {
    num_layers: usize,
    sizes: Vec<usize>,
    biases: Vec<Array2<f64>>,
    weights: Vec<Array2<f64>>,
    activation: A,
    loss: L,
}

impl<A: ActivationTrait, L: LossTrait> Network<A, L> {
       fn new(sizes: &[usize]) -> Self {

        let num_layers = sizes.len();
        let mut biases: Vec<Array2<f64>> = Vec::new();
        let mut weights: Vec<Array2<f64>> = Vec::new();

        for i in 1..num_layers {
            biases.push(Array::random((sizes[i], 1), Standard));
            weights.push(Array::random((sizes[i], sizes[i - 1]), Standard));
        }

        //Assume activation is relu and loss is crossentropy
        Self {
            num_layers: num_layers,
            sizes: sizes.to_owned(),
            biases: biases,
            weights: weights,
            activation: A::new(),
            loss: L::new(),
        }
    }

    fn predict(&self, n: f64) {
        //Just utilize activation function for now
        println!("Activation = {}", self.activation.activation(n));
    }

    fn update(&self, n: f64) {
        //Just utilize activation gradient function for now
        println!("Activation Gradient = {}", self.activation.activation_gradient(n));
    }
}

//Activation Functions
pub mod activation {
    use ndarray::Array2;

    pub trait ActivationTrait {
        fn activation(&self, z: f64) -> f64;
        fn activation_gradient(&self, z: f64) -> f64;
        fn activation_vectorized(&self, z: &mut Array2<f64>);
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
}

//Loss Functions
pub mod loss {
    //type MatrixVector = Array2<f64>; Just a thought to make code more readable, not necessary at all
    use ndarray::Array2;

    pub trait LossTrait {
        fn loss(&self, predicted: Array2<f64>, actual: Array2<f64>) -> f64;
        fn new() -> Self where Self: Sized;
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

        fn new() -> Self where Self: Sized {
            CrossEntropy
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

        fn new() -> Self where Self: Sized {
            Square
        }
    }
}

//Test Code
fn main() {
    //Create network
    let layers = [5, 4, 2];
    let n1 = Network::<Relu, CrossEntropy>::new(&layers); //Used "turbofish" syntax to make things cleaner
    let n2 = Network::<Sigmoid, Square>::new(&layers);

    println!("n1:");
    n1.predict(420.0);
    n1.update(420.0);
    n1.predict(-1.0);
    n1.update(-1.0);

    println!("\nn2:");
    n2.predict(0.4);
    n2.update(0.4);
    n2.predict(-1.0);
    n2.update(-1.0);

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
