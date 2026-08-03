// Memory layout of the values the hot paths pass around.
//
// ORDER BY and projection both funnel every cell through a `ScalarValue`. How
// big that enum is decides how much of a sort key set fits in cache, so the
// number is worth having in front of us rather than estimated.

use std::mem::size_of;

use falcondb::storage::types::ScalarValue;

#[test]
fn report_value_sizes() {
    let sv = size_of::<ScalarValue>();
    println!("size_of::<ScalarValue>()      = {sv} bytes");
    println!("size_of::<Option<f64>>()      = {} bytes", size_of::<Option<f64>>());
    println!("size_of::<f64>()              = {} bytes", size_of::<f64>());
    println!("size_of::<usize>()            = {} bytes", size_of::<usize>());
    println!("size_of::<(f64, u32)>()       = {} bytes", size_of::<(f64, u32)>());

    for rows in [100_000usize, 1_000_000] {
        println!(
            "\n{rows} rows: sort keys as Vec<ScalarValue> = {:.1} MB, \
             as Vec<(f64,u32)> = {:.1} MB, raw f64 = {:.1} MB",
            (rows * sv) as f64 / 1e6,
            (rows * size_of::<(f64, u32)>()) as f64 / 1e6,
            (rows * 8) as f64 / 1e6,
        );
    }
}
