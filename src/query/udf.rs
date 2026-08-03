// User-defined scalar functions.
//
// A UDF is a Rust closure callable from SQL by name:
//
//     falcondb::query::udf::register_udf("double", 1, |args| {
//         Ok(ScalarValue::Float64(args[0].as_f64().map(|v| v * 2.0)))
//     })?;
//     db.sql("SELECT double(score) FROM t")?;
//
// ## Why the registry is process-global
//
// UDFs are resolved inside `Expr::eval_batch`, which runs on rayon worker threads
// and takes no context parameter. Scoping the registry to a `Database` would mean
// threading a `&Context` through every expression evaluation call site and making
// it `Sync` — a large change for the benefit of isolating names.
//
// So the registry is global and shared by every `Database` in the process. Two
// consequences worth knowing:
//   * Registering the same name twice is an error rather than a silent overwrite,
//     so one part of a program cannot quietly change another's function.
//   * Tests that register UDFs must use distinct names, since they share the
//     registry and may run concurrently.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use crate::storage::types::{FalconError, FalconResult, ScalarValue};

/// A scalar function: one row of arguments in, one value out.
pub type ScalarFn = Arc<dyn Fn(&[&ScalarValue]) -> FalconResult<ScalarValue> + Send + Sync>;

/// A registered function together with the arity it expects.
#[derive(Clone)]
pub struct Udf {
    pub name:  String,
    /// Number of arguments required. `None` means variadic.
    pub arity: Option<usize>,
    pub func:  ScalarFn,
}

fn registry() -> &'static RwLock<HashMap<String, Udf>> {
    static REGISTRY: OnceLock<RwLock<HashMap<String, Udf>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Names that `eval_function` handles itself. A UDF may not take one of these:
/// built-ins are resolved first, so registering one would create a function that
/// is silently never called.
const BUILTIN_NAMES: &[&str] = &[
    "UPPER", "LOWER", "LENGTH", "LEN", "TRIM", "LTRIM", "RTRIM", "SUBSTR", "SUBSTRING",
    "CONCAT", "COALESCE", "NULLIF", "ABS", "ROUND", "CEIL", "CEILING", "FLOOR", "SQRT",
    "POW", "POWER", "SIGN", "IF", "YEAR", "MONTH", "DAY", "DAYOFMONTH", "DATEDIFF",
    "DATE_TRUNC", "NOW", "CURRENT_TIMESTAMP", "CURRENT_DATE", "TODAY",
];

/// Register a scalar UDF callable from SQL.
///
/// `arity` is the exact argument count; pass `None` for a variadic function.
/// Function names are matched case-insensitively, like built-ins.
///
/// Fails if the name collides with a built-in or with a UDF already registered —
/// use [`unregister_udf`] first to replace one deliberately.
pub fn register_udf<F>(name: &str, arity: impl Into<Option<usize>>, func: F) -> FalconResult<()>
where
    F: Fn(&[&ScalarValue]) -> FalconResult<ScalarValue> + Send + Sync + 'static,
{
    let key = name.to_uppercase();

    if key.is_empty() {
        return Err(FalconError::InvalidQuery("UDF name cannot be empty".into()));
    }
    if BUILTIN_NAMES.contains(&key.as_str()) {
        return Err(FalconError::InvalidQuery(format!(
            "'{name}' is a built-in function and cannot be redefined"
        )));
    }

    let mut reg = registry().write().unwrap();
    if reg.contains_key(&key) {
        return Err(FalconError::InvalidQuery(format!(
            "a function named '{name}' is already registered"
        )));
    }
    reg.insert(
        key,
        Udf { name: name.to_string(), arity: arity.into(), func: Arc::new(func) },
    );
    Ok(())
}

/// Remove a registered UDF. Returns whether one was present.
pub fn unregister_udf(name: &str) -> bool {
    registry().write().unwrap().remove(&name.to_uppercase()).is_some()
}

/// Names of all registered UDFs, in arbitrary order.
pub fn udf_names() -> Vec<String> {
    registry().read().unwrap().values().map(|u| u.name.clone()).collect()
}

/// Look up a UDF by name, case-insensitively.
pub fn lookup(name: &str) -> Option<Udf> {
    registry().read().unwrap().get(&name.to_uppercase()).cloned()
}

/// Invoke a registered UDF, checking arity first.
///
/// Returns `None` if no function of that name is registered, so the caller can
/// fall through to its own "unknown function" error.
pub fn call(name: &str, args: &[&ScalarValue]) -> Option<FalconResult<ScalarValue>> {
    let udf = lookup(name)?;

    if let Some(expected) = udf.arity {
        if args.len() != expected {
            return Some(Err(FalconError::InvalidQuery(format!(
                "{} expects {expected} argument(s), got {}",
                udf.name,
                args.len()
            ))));
        }
    }
    Some((udf.func)(args))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_call() {
        register_udf("udf_test_triple", 1, |args| {
            Ok(ScalarValue::Float64(args[0].as_f64().map(|v| v * 3.0)))
        })
        .unwrap();

        let arg = ScalarValue::Float64(Some(4.0));
        let got = call("udf_test_triple", &[&arg]).unwrap().unwrap();
        assert_eq!(got, ScalarValue::Float64(Some(12.0)));

        assert!(unregister_udf("udf_test_triple"));
    }

    #[test]
    fn name_lookup_is_case_insensitive() {
        register_udf("udf_test_Case", 0, |_| Ok(ScalarValue::Int64(Some(1)))).unwrap();
        assert!(lookup("UDF_TEST_CASE").is_some());
        assert!(lookup("udf_test_case").is_some());
        unregister_udf("udf_test_case");
    }

    #[test]
    fn arity_is_enforced() {
        register_udf("udf_test_arity", 2, |_| Ok(ScalarValue::Int64(Some(0)))).unwrap();
        let a = ScalarValue::Int64(Some(1));
        let err = call("udf_test_arity", &[&a]).unwrap().unwrap_err();
        assert!(err.to_string().contains("expects 2"));
        unregister_udf("udf_test_arity");
    }

    #[test]
    fn variadic_accepts_any_count() {
        register_udf("udf_test_variadic", None, |args| {
            Ok(ScalarValue::Int64(Some(args.len() as i64)))
        })
        .unwrap();
        let a = ScalarValue::Int64(Some(1));
        let got = call("udf_test_variadic", &[&a, &a, &a]).unwrap().unwrap();
        assert_eq!(got, ScalarValue::Int64(Some(3)));
        unregister_udf("udf_test_variadic");
    }

    #[test]
    fn builtins_cannot_be_shadowed() {
        let err = register_udf("upper", 1, |_| Ok(ScalarValue::Null)).unwrap_err();
        assert!(err.to_string().contains("built-in"));
    }

    #[test]
    fn duplicate_registration_is_rejected() {
        register_udf("udf_test_dup", 0, |_| Ok(ScalarValue::Null)).unwrap();
        assert!(register_udf("udf_test_dup", 0, |_| Ok(ScalarValue::Null)).is_err());
        unregister_udf("udf_test_dup");
    }

    #[test]
    fn unknown_name_returns_none() {
        assert!(call("udf_test_definitely_not_registered", &[]).is_none());
    }
}
