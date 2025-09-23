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
        .finish();
    println!("{:?}", df);
}
