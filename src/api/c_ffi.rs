// C FFI — safe wrappers around Database for use from C, Python (ctypes), etc.
//
// Usage from C:
//   falcondb_db* db = falcondb_open();
//   falcondb_load_csv(db, "data.csv", "users");
//   falcondb_result* r = falcondb_sql(db, "SELECT * FROM users LIMIT 10");
//   printf("%s\n", falcondb_result_display(r));
//   falcondb_result_free(r);
//   falcondb_close(db);

use std::{
    ffi::{CStr, CString},
    os::raw::{c_char, c_int, c_ulonglong},
    ptr,
};

use crate::db::{Database, QueryResult};

// ─── Opaque handles ───────────────────────────────────────────────────────────

pub struct FalconDb(Database);
pub struct FalconResult(QueryResult);

// ─── Database lifecycle ───────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn falcondb_open() -> *mut FalconDb {
    Box::into_raw(Box::new(FalconDb(Database::in_memory())))
}

#[no_mangle]
pub unsafe extern "C" fn falcondb_close(db: *mut FalconDb) {
    if !db.is_null() {
        drop(Box::from_raw(db));
    }
}

// ─── Data loading ─────────────────────────────────────────────────────────────

/// Returns number of rows loaded, or -1 on error.
#[no_mangle]
pub unsafe extern "C" fn falcondb_load_csv(
    db:         *mut FalconDb,
    path:       *const c_char,
    table_name: *const c_char,
) -> c_int {
    if db.is_null() || path.is_null() || table_name.is_null() {
        return -1;
    }
    let path_str  = CStr::from_ptr(path).to_string_lossy();
    let table_str = CStr::from_ptr(table_name).to_string_lossy();

    match (*db).0.load_csv(path_str.as_ref(), table_str.as_ref()) {
        Ok(n)  => n as c_int,
        Err(_) => -1,
    }
}

/// Returns number of rows loaded, or -1 on error.
#[no_mangle]
pub unsafe extern "C" fn falcondb_load_fdb(
    db:         *mut FalconDb,
    path:       *const c_char,
    table_name: *const c_char,
) -> c_int {
    if db.is_null() || path.is_null() || table_name.is_null() {
        return -1;
    }
    let path_str  = CStr::from_ptr(path).to_string_lossy();
    let table_str = CStr::from_ptr(table_name).to_string_lossy();

    match (*db).0.load_fdb(path_str.as_ref(), table_str.as_ref()) {
        Ok(n)  => n as c_int,
        Err(_) => -1,
    }
}

// ─── Query ────────────────────────────────────────────────────────────────────

/// Execute SQL. Returns a heap-allocated result; free with falcondb_result_free.
/// Returns NULL on error.
#[no_mangle]
pub unsafe extern "C" fn falcondb_sql(
    db:  *mut FalconDb,
    sql: *const c_char,
) -> *mut FalconResult {
    if db.is_null() || sql.is_null() {
        return ptr::null_mut();
    }
    let sql_str = CStr::from_ptr(sql).to_string_lossy();

    match (*db).0.sql(sql_str.as_ref()) {
        Ok(r)  => Box::into_raw(Box::new(FalconResult(r))),
        Err(_) => ptr::null_mut(),
    }
}

// ─── Result accessors ─────────────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn falcondb_result_row_count(r: *const FalconResult) -> c_ulonglong {
    if r.is_null() { return 0; }
    (*r).0.row_count() as c_ulonglong
}

/// Returns a pretty-printed table as a null-terminated C string.
/// The returned pointer is valid until the result is freed.
/// Caller must NOT free this string directly.
#[no_mangle]
pub unsafe extern "C" fn falcondb_result_display(r: *mut FalconResult) -> *const c_char {
    if r.is_null() { return ptr::null(); }
    let s = (*r).0.display();
    match CString::new(s) {
        Ok(cs) => {
            let ptr = cs.as_ptr();
            // Leak here intentionally — the string lives as long as the result.
            std::mem::forget(cs);
            ptr
        }
        Err(_) => ptr::null(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn falcondb_result_free(r: *mut FalconResult) {
    if !r.is_null() {
        drop(Box::from_raw(r));
    }
}

// ─── Table metadata ───────────────────────────────────────────────────────────

/// Returns row count for the named table, or 0 if not found.
#[no_mangle]
pub unsafe extern "C" fn falcondb_table_row_count(
    db:   *const FalconDb,
    name: *const c_char,
) -> c_ulonglong {
    if db.is_null() || name.is_null() { return 0; }
    let n = CStr::from_ptr(name).to_string_lossy();
    (*db).0.table_info(n.as_ref())
        .map(|i| i.row_count as c_ulonglong)
        .unwrap_or(0)
}
