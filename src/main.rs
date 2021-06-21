use rand::distributions::Standard;
use ndarray::{Array, Array2};
use ndarray_rand::RandomExt;

#[derive(Debug)]
struct Network {
    num_layers: usize,
    sizes: Vec<usize>,
    biases: Vec<Array2<f64>>,
    weights: Vec<Array2<f64>>,
}


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
}

fn main() {
    
}