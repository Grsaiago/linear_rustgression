use polars::prelude::*;

fn main() {
    println!("Hello, world!");
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
        let lf = df!(
            "km" => [
                240000, 139800, 150500, 185530, 176000, 114800, 166800, 89000, 144500
            ],
            "price" => [
                3650, 3800, 4400, 4450, 5250, 5350, 5800, 5990, 5999
            ],
        )
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
