// Does the index answer the same question as a full scan?
//
// An index is only ever an optimisation: for any predicate it accepts, it must
// return exactly what scanning the table would. `execute_select_with_index`
// treats an empty index lookup as a final answer —
//
//     if hits.is_empty() { return Some(Vec::new()); }
//
// — so an index that fails to match a row does not fall back, it reports zero
// rows. Any gap between "the index found nothing" and "there is nothing" is a
// silent wrong answer rather than a slow query.
//
// The suspicion under test: `IndexKey` sorts integers into `Int` and floats
// into `Float`, which are different enum variants and therefore different
// BTreeMap keys. If `WHERE score = 5` on a DOUBLE column produces an integer
// literal, the probe key is `Int(5)` while every key in the tree is `Float(..)`,
// and the lookup misses.
//
// Every test here asks the same query twice — once with the index, once on an
// identical table without one — and compares. That way the assertion is
// "the index agrees with a scan", which stays meaningful no matter which
// literal type the parser happens to produce.

use falcondb::db::Database;

/// Build two identical tables; only `indexed` gets an index on `col`.
fn twin(ddl: &str, rows: &[&str], col: &str) -> Database {
    let db = Database::in_memory();
    for name in ["indexed", "plain"] {
        db.sql(&ddl.replace("{}", name)).unwrap();
        for r in rows {
            db.sql(&format!("INSERT INTO {name} VALUES ({r})")).unwrap();
        }
    }
    db.sql(&format!("CREATE INDEX ix ON indexed ({col})")).unwrap();
    db
}

/// Run `query` against both twins and assert they agree.
fn agree(db: &Database, query: &str) -> usize {
    let with = db.sql(&query.replace("{}", "indexed")).unwrap();
    let without = db.sql(&query.replace("{}", "plain")).unwrap();

    let mut a = with.to_rows();
    let mut b = without.to_rows();
    a.sort();
    b.sort();
    assert_eq!(
        a, b,
        "index disagrees with a full scan for: {query}\n  indexed -> {a:?}\n  scan    -> {b:?}"
    );
    a.len()
}

#[test]
fn integer_literal_against_a_double_column() {
    // The suspected case: the column holds floats, the literal is written
    // without a decimal point.
    let db = twin(
        "CREATE TABLE {} (score DOUBLE)",
        &["1.0", "5.0", "9.0"],
        "score",
    );
    assert_eq!(agree(&db, "SELECT score FROM {} WHERE score = 5"), 1);
}

#[test]
fn float_literal_against_a_double_column() {
    let db = twin(
        "CREATE TABLE {} (score DOUBLE)",
        &["1.0", "5.0", "9.0"],
        "score",
    );
    assert_eq!(agree(&db, "SELECT score FROM {} WHERE score = 5.0"), 1);
}

#[test]
fn float_literal_against_an_integer_column() {
    // The mirror image: integer column, literal written as a float.
    let db = twin("CREATE TABLE {} (x INT)", &["1", "5", "9"], "x");
    agree(&db, "SELECT x FROM {} WHERE x = 5.0");
}

#[test]
fn integer_literal_against_an_integer_column() {
    // The covered case, kept as the control: if this one fails the harness is
    // wrong, not the index.
    let db = twin("CREATE TABLE {} (x INT)", &["1", "5", "9"], "x");
    assert_eq!(agree(&db, "SELECT x FROM {} WHERE x = 5"), 1);
}

#[test]
fn range_over_a_double_column_with_integer_bounds() {
    let db = twin(
        "CREATE TABLE {} (score DOUBLE)",
        &["1.5", "3.5", "5.5", "7.5"],
        "score",
    );
    agree(&db, "SELECT score FROM {} WHERE score BETWEEN 3 AND 6");
}

#[test]
fn range_over_a_double_column_with_float_bounds() {
    let db = twin(
        "CREATE TABLE {} (score DOUBLE)",
        &["1.5", "3.5", "5.5", "7.5"],
        "score",
    );
    assert_eq!(
        agree(&db, "SELECT score FROM {} WHERE score BETWEEN 3.0 AND 6.0"),
        2
    );
}

#[test]
fn equality_on_a_text_column() {
    let db = twin(
        "CREATE TABLE {} (name TEXT)",
        &["'alice'", "'bob'"],
        "name",
    );
    assert_eq!(agree(&db, "SELECT name FROM {} WHERE name = 'bob'"), 1);
}

#[test]
fn a_value_that_is_genuinely_absent_returns_nothing_either_way() {
    // The other direction: the index must not invent rows a scan would not find.
    let db = twin("CREATE TABLE {} (x INT)", &["1", "5", "9"], "x");
    assert_eq!(agree(&db, "SELECT x FROM {} WHERE x = 4"), 0);
}
