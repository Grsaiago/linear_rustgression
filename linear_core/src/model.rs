use polars::prelude::*;

struct SlopeIntercept {
    slope: f64,
    intercept: f64,
}

pub struct LinearRegressionModel {
    inner: SlopeIntercept,
}

impl LinearRegressionModel {
    pub fn new_untrained() -> LinearRegressionModel {
        LinearRegressionModel {
            inner: SlopeIntercept {
                slope: 0.0,
                intercept: 0.0,
            },
        }
    }

    pub fn new(slope: f64, intercept: f64) -> LinearRegressionModel {
        LinearRegressionModel {
            inner: SlopeIntercept { slope, intercept },
        }
    }

    pub fn slope(&self) -> f64 {
        self.inner.slope
    }

    pub fn intercept(&self) -> f64 {
        self.inner.intercept
    }

    /// é um jeito bonito de 'buscar um ponto na nossa reta'
    /// usando a equação geral da reta y = ax + b
    pub fn predict(&self, x: f64) -> f64 {
        (self.slope() * x) + self.intercept()
    }

    // Um gradiente descendente com a função de custo MSE (Mean Squared Error)
    pub fn train(&mut self, mileage: Series, price: Series, learning_rate: f64) {
        let max_iteration = 1000;

        let new_values = (0..max_iteration).fold(
            SlopeIntercept {
                slope: 0.0,
                intercept: 0.0,
            },
            |curr, _| {
                let current_prediction: Series = &mileage * curr.slope + curr.intercept;

                // mean because we iterate and sum intil number_of_elements and then
                // divide by number_of_elements, so, a mean
                let temp_intercept =
                    (&current_prediction - &price).unwrap().mean().unwrap() * learning_rate;

                // mean because we iterate and sum intil number_of_elements and then
                // divide by number_of_elements, so, a mean
                let temp_slope = (&current_prediction - &price)
                    .unwrap()
                    .multiply(&mileage)
                    .unwrap()
                    .mean()
                    .unwrap()
                    * learning_rate;

                SlopeIntercept {
                    slope: curr.slope - temp_slope,
                    intercept: curr.intercept - temp_intercept,
                }
            },
        );

        self.inner.slope = new_values.slope;
        self.inner.intercept = new_values.intercept;
    }
}
