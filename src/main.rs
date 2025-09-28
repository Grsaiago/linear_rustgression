use polars::prelude::*;
use std::path::PathBuf;

struct SlopeIntercept {
    slope: f64,
    intercept: f64,
}

struct LinearRegressionModel {
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
        // iterative
        // for _ in 0..max_iteration {
        //     let current_prediction: Series = &mileage * slope + intercept;
        //
        //     // mean because we iterate and sum intil number_of_elements and then
        //     // divide by number_of_elements, so, a mean
        //     let temp_intercept =
        //         (&current_prediction - &price).unwrap().mean().unwrap() * learning_rate;
        //
        //     // mean because we iterate and sum intil number_of_elements and then
        //     // divide by number_of_elements, so, a mean
        //     let temp_slope = (&current_prediction - &price)
        //         .unwrap()
        //         .multiply(&mileage)
        //         .unwrap()
        //         .mean()
        //         .unwrap()
        //         * learning_rate;
        //     slope = slope - temp_slope;
        //     intercept = intercept - temp_intercept;
        // }
        // SlopeIntercept::new(slope, intercept)
    }
}

fn main() {
    let lf = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/",
            "dataset.csv"
        ))))
        .unwrap()
        .finish()
        .unwrap()
        .lazy();

    let mut z_df = z_normalize_dataframe(&lf);

    let normalized_price: Series = z_df
        .drop_in_place("z_price")
        .unwrap()
        .take_materialized_series();
    let normalized_mileage = z_df
        .drop_in_place("z_km")
        .unwrap()
        .take_materialized_series();

    let mut model = LinearRegressionModel::new_untrained();

    model.train(normalized_mileage, normalized_price, 0.01);

    println!("{:?}", z_df);
}

fn gradient_descent(
    mileage: Series,
    price: Series,
    learning_rate: f64,
    max_iteration: u16,
) -> LinearRegressionModel {
    (0..max_iteration).fold(LinearRegressionModel::new(0.0, 0.0), |curr, _| {
        let current_prediction: Series = &mileage * curr.slope() + curr.intercept();

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

        LinearRegressionModel::new(curr.slope() - temp_slope, curr.intercept() - temp_intercept)
    })
    // iterative
    // for _ in 0..max_iteration {
    //     let current_prediction: Series = &mileage * slope + intercept;
    //
    //     // mean because we iterate and sum intil number_of_elements and then
    //     // divide by number_of_elements, so, a mean
    //     let temp_intercept =
    //         (&current_prediction - &price).unwrap().mean().unwrap() * learning_rate;
    //
    //     // mean because we iterate and sum intil number_of_elements and then
    //     // divide by number_of_elements, so, a mean
    //     let temp_slope = (&current_prediction - &price)
    //         .unwrap()
    //         .multiply(&mileage)
    //         .unwrap()
    //         .mean()
    //         .unwrap()
    //         * learning_rate;
    //     slope = slope - temp_slope;
    //     intercept = intercept - temp_intercept;
    // }
    // SlopeIntercept::new(slope, intercept)
}

fn z_normalize_dataframe(lf: &LazyFrame) -> DataFrame {
    let mut lf = lf.clone();

    let lf_schema = lf.collect_schema().unwrap();

    lf.select(
        lf_schema
            .iter_names()
            .map(|column_name| {
                ((col(column_name.clone()) - col(column_name.clone()).mean())
                    / col(column_name.clone()).std(1))
                .alias(format!("z_{column_name}"))
            })
            .collect::<Vec<_>>(),
    )
    .collect()
    .unwrap()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_z_normalize_dataframe() {
        let lf = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some(PathBuf::from(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/",
                "dataset.csv"
            ))))
            .unwrap()
            .finish()
            .unwrap()
            .lazy();
        let normalized_df = z_normalize_dataframe(&lf);
        let check = normalized_df
            .clone()
            .lazy()
            .select([
                col("z_price")
                    .mean()
                    .round(2, RoundMode::HalfAwayFromZero)
                    .alias("mean_price"),
                col("z_price")
                    .std(1)
                    .alias("std_price")
                    .round(2, RoundMode::HalfAwayFromZero),
                col("z_km")
                    .mean()
                    .round(2, RoundMode::HalfAwayFromZero)
                    .alias("mean_km"),
                col("z_km")
                    .std(1)
                    .round(2, RoundMode::HalfAwayFromZero)
                    .alias("std_km"),
            ])
            .collect()
            .unwrap();
        let expected = df![
            "mean_price" => [0.0],
            "std_price" => [1.0],
            "mean_km" => [0.0],
            "std_km" => [1.0],
        ]
        .unwrap();
        assert!(check.equals(&expected));
    }
}
