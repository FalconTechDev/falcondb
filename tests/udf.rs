// User-defined scalar functions, exercised through SQL.
//
// The registry is process-global (see src/query/udf.rs for why), and tests in a
// binary run concurrently by default, so every test here uses a name unique to
// itself rather than relying on registration order.

use std::io::Write;

use falcondb::{
    db::Database,
    query::udf::{register_udf, udf_names, unregister_udf},
    storage::types::{FalconError, ScalarValue},
};
use tempfile::NamedTempFile;

fn db_with_scores() -> (Database, NamedTempFile) {
    let mut f = NamedTempFile::new().unwrap();
    write!(f, "id,score,name\n1,10.0,alice\n2,20.0,bob\n3,30.0,carol\n").unwrap();
    f.flush().unwrap();
    let db = Database::in_memory();
    db.load_csv(f.path(), "t").unwrap();
    (db, f)
}

fn col0(db: &Database, sql: &str) -> Vec<String> {
    db.sql(sql).unwrap().to_rows().into_iter().map(|r| r[0].clone()).collect()
}

#[test]
fn udf_is_callable_from_sql() {
    register_udf("t_double", 1, |args| {
        Ok(ScalarValue::Float64(args[0].as_f64().map(|v| v * 2.0)))
    })
    .unwrap();

    let (db, _f) = db_with_scores();
    let got = col0(&db, "SELECT t_double(score) FROM t ORDER BY id");
    assert_eq!(got, vec!["20", "40", "60"]);

    unregister_udf("t_double");
}

#[test]
fn udf_works_in_a_where_clause() {
    register_udf("t_over_fifteen", 1, |args| {
        Ok(ScalarValue::Boolean(args[0].as_f64().map(|v| v > 15.0)))
    })
    .unwrap();

    let (db, _f) = db_with_scores();
    let got = col0(&db, "SELECT id FROM t WHERE t_over_fifteen(score) ORDER BY id");
    assert_eq!(got, vec!["2", "3"]);

    unregister_udf("t_over_fifteen");
}

#[test]
fn udf_can_take_strings() {
    register_udf("t_initial", 1, |args| {
        Ok(match args[0] {
            ScalarValue::Utf8(Some(s)) => {
                ScalarValue::Utf8(s.chars().next().map(|c| c.to_string()))
            }
            _ => ScalarValue::Null,
        })
    })
    .unwrap();

    let (db, _f) = db_with_scores();
    let got = col0(&db, "SELECT t_initial(name) FROM t ORDER BY id");
    assert_eq!(got, vec!["a", "b", "c"]);

    unregister_udf("t_initial");
}

#[test]
fn udf_can_take_several_arguments() {
    register_udf("t_clamp", 3, |args| {
        let (v, lo, hi) = (args[0].as_f64(), args[1].as_f64(), args[2].as_f64());
        Ok(match (v, lo, hi) {
            (Some(v), Some(lo), Some(hi)) => ScalarValue::Float64(Some(v.clamp(lo, hi))),
            _ => ScalarValue::Null,
        })
    })
    .unwrap();

    let (db, _f) = db_with_scores();
    let got = col0(&db, "SELECT t_clamp(score, 15, 25) FROM t ORDER BY id");
    assert_eq!(got, vec!["15", "20", "25"]);

    unregister_udf("t_clamp");
}

#[test]
fn udf_composes_with_builtins() {
    register_udf("t_shout", 1, |args| {
        Ok(match args[0] {
            ScalarValue::Utf8(Some(s)) => ScalarValue::Utf8(Some(format!("{s}!"))),
            _ => ScalarValue::Null,
        })
    })
    .unwrap();

    let (db, _f) = db_with_scores();
    let got = col0(&db, "SELECT UPPER(t_shout(name)) FROM t ORDER BY id");
    assert_eq!(got, vec!["ALICE!", "BOB!", "CAROL!"]);

    unregister_udf("t_shout");
}

#[test]
fn udf_survives_aggregation() {
    register_udf("t_half", 1, |args| {
        Ok(ScalarValue::Float64(args[0].as_f64().map(|v| v / 2.0)))
    })
    .unwrap();

    let (db, _f) = db_with_scores();
    let got = col0(&db, "SELECT SUM(t_half(score)) AS s FROM t");
    assert_eq!(got[0].parse::<f64>().unwrap(), 30.0);

    unregister_udf("t_half");
}

#[test]
fn udf_errors_propagate_to_the_caller() {
    register_udf("t_always_fails", 1, |_| {
        Err(FalconError::InvalidQuery("deliberate failure".into()))
    })
    .unwrap();

    let (db, _f) = db_with_scores();
    let err = match db.sql("SELECT t_always_fails(score) FROM t") {
        Err(e) => e.to_string(),
        Ok(_) => panic!("a failing UDF should surface its error"),
    };
    assert!(err.contains("deliberate failure"), "got: {err}");

    unregister_udf("t_always_fails");
}

#[test]
fn wrong_argument_count_is_reported() {
    register_udf("t_needs_two", 2, |_| Ok(ScalarValue::Null)).unwrap();

    let (db, _f) = db_with_scores();
    let err = match db.sql("SELECT t_needs_two(score) FROM t") {
        Err(e) => e.to_string(),
        Ok(_) => panic!("arity mismatch should be an error"),
    };
    assert!(err.contains("expects 2"), "got: {err}");

    unregister_udf("t_needs_two");
}

#[test]
fn unknown_function_still_errors() {
    let (db, _f) = db_with_scores();
    let err = match db.sql("SELECT t_never_registered(score) FROM t") {
        Err(e) => e.to_string(),
        Ok(_) => panic!("an unregistered function should not resolve"),
    };
    assert!(err.contains("unknown function"), "got: {err}");
}

#[test]
fn unregistering_removes_it_from_sql() {
    register_udf("t_temporary", 1, |args| Ok((*args[0]).clone())).unwrap();
    let (db, _f) = db_with_scores();
    assert!(db.sql("SELECT t_temporary(score) FROM t").is_ok());

    assert!(unregister_udf("t_temporary"));
    assert!(db.sql("SELECT t_temporary(score) FROM t").is_err());
}

#[test]
fn registered_names_are_listed() {
    register_udf("t_listed", 0, |_| Ok(ScalarValue::Null)).unwrap();
    assert!(udf_names().iter().any(|n| n == "t_listed"));
    unregister_udf("t_listed");
}

#[test]
fn udf_handles_nulls() {
    register_udf("t_is_null_marker", 1, |args| {
        Ok(ScalarValue::Utf8(Some(
            if args[0].is_null() { "none" } else { "some" }.to_string(),
        )))
    })
    .unwrap();

    let mut f = NamedTempFile::new().unwrap();
    write!(f, "id,v\n1,5\n2,\n").unwrap();
    f.flush().unwrap();
    let db = Database::in_memory();
    db.load_csv(f.path(), "t").unwrap();

    let got = col0(&db, "SELECT t_is_null_marker(v) FROM t ORDER BY id");
    assert_eq!(got, vec!["some", "none"]);

    unregister_udf("t_is_null_marker");
}
