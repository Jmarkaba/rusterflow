use ndarray::Array2;
use crate::layer::*;
use crate::loss::*;

pub struct Datum {
    pub input: Array2<f64>,
    pub actual: Array2<f64>,
}

type Batch = Vec<Datum>;

pub struct Network<L: Loss, Y: Layer> {
    num_layers: usize,
    layers: Vec<Y>,
    loss: L,
}

impl<L: Loss, Y: Layer> Network<L, Y> {
    pub fn new(loss: L) -> Self {
        Self {
            num_layers: 0,
            layers: Vec::new(),
            loss: L::new()
        }
    }

    //Could possibly change to dyn Layer
    pub fn add(&mut self, layer: Y) {
        self.layers.push(layer);
        self.num_layers += 1;
    }

    pub fn predict(&mut self, input: &Array2<f64>) -> Array2<f64> {
        let mut x = input.clone();

        for layer in &mut self.layers {
            x = layer.forward(&x);
        }

        x
    }

    fn update(&mut self, datum: &Datum, learning_rate: f64) {
        let predicted = self.predict(&datum.input);
        let loss = self.loss.value(&predicted, &datum.actual);
        let loss_gradient = self.loss.gradient(&predicted, &datum.actual);
        let mut dJ_da = loss_gradient;

        for layer in &mut self.layers.iter_mut().rev() {
            dJ_da = layer.backward(&dJ_da);
        }

        for layer in &mut self.layers {
            layer.update(learning_rate);
        }
    }

    pub fn batch_update(&mut self, batch: &Batch, learning_rate: f64) {
        for datum in batch {
            self.update(datum, learning_rate / (batch.len() as f64));
        }
    }
}
