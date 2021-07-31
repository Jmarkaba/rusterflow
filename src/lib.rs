mod network;
mod activation;
mod layer;
mod loss;

#[cfg(test)]
mod tests {
    use crate::loss::*;
    use crate::network::*;
    use crate::layer::*;
    use crate::activation::*;
    use crate::loss::*;
    use ndarray::array;

    #[test]
    fn test_predict() {
        let mut n: Network<Square, DenseLayer<Linear>> = Network::new(Square::new());

        n.add(DenseLayer::new(array![[1]], array![[1]], Linear::new()));
        n.add(DenseLayer::new(array![[1]], array![[1]], Linear::new()));
        n.add(DenseLayer::new(array![[1]], array![[1]], Linear::new()));

        let mut prediction = n.predict(&array![[2.0]]);
        println!("Initial Prediction: {}", prediction);

        //3x+1
        let mut batch = Vec::new();

        for i in 1..50 {
            let x = i as f64;
            let datum = Datum {
                input: array![[x]],
                actual: array![[3.0 * x + 1.0]],
            };

            batch.push(datum);
        }

        for _ in 1..1000 {
            n.batch_update(&batch, 0.1);
        }

        prediction = n.predict(&array![[10.0]]);
        println!("Updated Prediction: {}", prediction);
    }
}
