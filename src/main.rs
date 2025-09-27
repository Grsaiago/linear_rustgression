use polars::prelude::*;
use std::path::PathBuf;

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
