use polars::prelude::*;
use std::path::PathBuf;

fn main() {
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/",
            "dataset.csv"
        ))))
        .unwrap()
        .finish()
        .unwrap();

    let km_series = df.column("km").unwrap().as_series().unwrap();
    let price_series = df.column("price").unwrap().as_series().unwrap();

    let km_mean = km_series.mean().unwrap();
    let km_stddev = km_series.std(1).unwrap();

    let price_mean = price_series.mean().unwrap();
    let price_stddev = price_series.std(1).unwrap();

    println!("km_mean: {km_mean:?}");
    println!("km_stddv: {km_stddev:?}");

    println!("price_mean: {price_mean:?}");
    println!("price_stddv: {price_stddev:?}");
    println!("{df:?}");

    let normalized_df = df!(
        "km" => km_series.i64().unwrap().iter().map(|val| {
            (val.unwrap() as f64 * km_mean) / km_stddev
    }).collect::<Vec<_>>(),
        "price" => price_series.i64().unwrap().iter().map(|val| {
            (val.unwrap() as f64 * price_mean) / price_stddev
        }).collect::<Vec<_>>()
    );
    println!("{normalized_df:?}");
}

fn get_z_normalized_series(
    series: &polars::series::Series,
    datatype: polars::datatypes::DataType,
) -> impl Into<Series> {
    let stddev: f64 = series.std_reduce(1).unwrap().as_any_value().strict_cast
    let mean = series.mean().unwrap();

    match datatype {
        DataType::Int64 => series
            .i64()
            .unwrap()
            .iter()
            .map(|val| (val.unwrap() as f64 * mean) / stddev)
            .collect::<Series>(),
        DataType::Float64 => series
            .f64()
            .unwrap()
            .iter()
            .map(|val| (val.unwrap() as f64 * mean) / stddev)
            .collect::<Series>(),
        _ => panic!("unsuported value type"),
    }
}
