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
    fn predict_linear_regression() {
        let mut n: Network<Square, DenseLayer<Linear>> = Network::new(Square::new());

        n.add(DenseLayer::new(array![[2]], array![[2]], Linear::new()));
        n.add(DenseLayer::new(array![[2]], array![[1]], Linear::new()));
        //n.add(DenseLayer::new(array![[1]], array![[1]], Linear::new()));

        let mut prediction = n.predict(&array![[12.0, 35.0]]);
        println!("Initial Prediction: {} | Actual: {}", prediction, 2.0 * 12.0 + 3.0 * 35.0 + 12.0);

        let mut batch = Vec::new();

        //2x + 3y + 12
        for i in 1..50 {
            for j in 1..50 {
                let x = i as f64;
                let y = j as f64;
                let datum = Datum {
                    input: array![[x, y]],
                    actual: array![[2.0 * x + 3.0 * y + 12.0]],
                };

                //DEBUG
                //println!("input shape: {}, {}", datum.input.dim().0, datum.input.dim().1);

                batch.push(datum);
            }
        }

        for i in 1..501 {
            n.batch_update(&batch, 0.000005);
            println!("EPOCH: {} | LOSS: {}", i, n.batch_loss(&batch));

            if i % 10 == 0 {
                prediction = n.predict(&array![[12.0, 35.0]]);
                println!("Updated Prediction: {} | Actual: {}", prediction, 2.0 * 12.0 + 3.0 * 35.0 + 12.0);
            }
        }
    }

    //#[test]
    fn predict_quadratic() {
        let mut n: Network<Square, DenseLayer<Relu>> = Network::new(Square::new());

        n.add(DenseLayer::new(array![[1]], array![[1]], Relu::new()));
        n.add(DenseLayer::new(array![[1]], array![[1]], Relu::new()));
        n.add(DenseLayer::new(array![[1]], array![[1]], Relu::new()));

        let mut prediction = n.predict(&array![[3.0]]);
        println!("Initial Prediction: {} | Actual: {}", prediction, 3.0 * 3.0 * 3.0 + 2.0 * 3.0 + 1.0);

        let mut batch = Vec::new();

        //3x^2 + 2x + 1
        for i in 1..50 {
            for j in 1..50 {
                let x = i as f64;
                let y = j as f64;
                let datum = Datum {
                    input: array![[x]],
                    actual: array![[3.0 * x * x + 2.0 * x + 1.0]],
                };

                //DEBUG
                //println!("input shape: {}, {}", datum.input.dim().0, datum.input.dim().1);

                batch.push(datum);
            }
        }

        for i in 1..201 {
            n.batch_update(&batch, 0.00000000001);
            println!("EPOCH: {} | LOSS: {}", i, n.batch_loss(&batch));

            if i % 10 == 0 {
                prediction = n.predict(&array![[3.0]]);
                println!("Updated Prediction: {} | Actual: {}", prediction, 3.0 * 3.0 * 3.0 + 2.0 * 3.0 + 1.0);
            }
        }
    }
}
