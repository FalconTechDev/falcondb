// Python bindings for FalconDB using PyO3.
//
// Build with:  maturin develop --features python
// or:          cargo build --features python
//
// Usage from Python:
//   import falcondb
//   db = falcondb.Database()
//   db.sql("CREATE TABLE t (id INT, name TEXT)")
//   db.sql("INSERT INTO t VALUES (1, 'alice')")
//   result = db.sql("SELECT * FROM t")
//   print(result.rows())

#![cfg(feature = "python")]

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::Arc;

use crate::db::Database as InnerDatabase;

// ─── PyDatabase ───────────────────────────────────────────────────────────────

#[pyclass(name = "Database")]
pub struct PyDatabase {
    inner: Arc<InnerDatabase>,
}

#[pymethods]
impl PyDatabase {
    #[new]
    fn new() -> Self {
        Self { inner: Arc::new(InnerDatabase::in_memory()) }
    }

    /// Execute any SQL statement. Returns a QueryResult object.
    fn sql(&self, py: Python<'_>, query: &str) -> PyResult<PyQueryResult> {
        self.inner
            .sql(query)
            .map(|r| PyQueryResult { columns: r.column_names(), rows: r.to_rows() })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Load a CSV file into a named table. Returns the number of rows loaded.
    fn load_csv(&self, path: &str, table: &str) -> PyResult<usize> {
        self.inner
            .load_csv(path, table)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// List all table names.
    fn table_names(&self) -> Vec<String> {
        self.inner.table_names()
    }
}

// ─── PyQueryResult ────────────────────────────────────────────────────────────

#[pyclass(name = "QueryResult")]
pub struct PyQueryResult {
    columns: Vec<String>,
    rows:    Vec<Vec<String>>,
}

#[pymethods]
impl PyQueryResult {
    /// Column names as a list of strings.
    fn columns(&self) -> Vec<String> {
        self.columns.clone()
    }

    /// Rows as a list of lists of strings.
    fn rows(&self) -> Vec<Vec<String>> {
        self.rows.clone()
    }

    /// Row count.
    fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Return rows as a list of dicts (column name → value).
    fn to_dicts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for row in &self.rows {
            let d = PyDict::new(py);
            for (col, val) in self.columns.iter().zip(row.iter()) {
                d.set_item(col, val)?;
            }
            list.append(d)?;
        }
        Ok(list)
    }

    fn __repr__(&self) -> String {
        format!(
            "QueryResult({} rows × {} cols)",
            self.rows.len(),
            self.columns.len()
        )
    }
}

// ─── Module registration ──────────────────────────────────────────────────────

#[pymodule]
fn falcondb(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDatabase>()?;
    m.add_class::<PyQueryResult>()?;
    Ok(())
}
