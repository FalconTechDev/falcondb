use std::{
    collections::{HashMap, HashSet},
    path::Path,
    sync::{atomic::{AtomicBool, Ordering}, Arc, Mutex, RwLock},
};

use crate::{
    query::{
        aggregate::execute_agg_only,

        select::{apply_offset_limit, display_results, execute_select, project_batch, Projection},
        sql::parse_sql,
    },
    storage::{
        csv::{read_csv, read_csv_with_options, CsvReadOptions},
        native::{read_table, write_table},
        table::{RecordBatch, Table},
        types::{FalconError, FalconResult},
        wal::{Wal, WalEntryType},
    },
};

// ─── QueryResult ─────────────────────────────────────────────────────────────

pub struct QueryResult {
    pub batches: Vec<RecordBatch>,
}

impl QueryResult {
    pub fn row_count(&self) -> usize {
        self.batches.iter().map(|b| b.row_count()).sum()
    }

    pub fn column_names(&self) -> Vec<String> {
        self.batches
            .first()
            .map(|b| b.schema.column_names().into_iter().map(|s| s.to_string()).collect())
            .unwrap_or_default()
    }

    /// Return rows as Vec<Vec<String>> — each inner vec is one row of display values.
    pub fn to_rows(&self) -> Vec<Vec<String>> {
        let col_count = self.batches.first()
            .map(|b| b.column_count())
            .unwrap_or(0);

        let mut rows = Vec::new();
        for batch in &self.batches {
            for row_idx in 0..batch.row_count() {
                let row: Vec<String> = (0..col_count)
                    .map(|ci| {
                        batch.column(ci).get(row_idx)
                            .map(|v| v.to_string())
                            .unwrap_or_else(|_| "ERR".into())
                    })
                    .collect();
                rows.push(row);
            }
        }
        rows
    }

    pub fn display(&self) -> String {
        display_results(&self.batches)
    }
}

// ─── Transaction state ────────────────────────────────────────────────────────

struct TransactionState {
    /// Snapshot of all tables at BEGIN time; used to restore on ROLLBACK.
    snapshot:   HashMap<String, Arc<Table>>,
    /// WAL entries buffered during the transaction; flushed atomically on COMMIT.
    wal_buffer: Vec<(WalEntryType, String, String)>,
}

// ─── Database ────────────────────────────────────────────────────────────────

/// Thread-safe database instance. Holds tables in memory and a WAL for durability.
pub struct Database {
    tables:      RwLock<HashMap<String, Arc<Table>>>,
    wal:         Option<RwLock<Wal>>,
    plan_cache:  RwLock<HashMap<String, Arc<crate::query::sql::ParsedQuery>>>,
    /// Set while `recover` is running so replayed statements are not re-journalled.
    replaying:   AtomicBool,
    /// Active transaction state, or None when outside a transaction.
    transaction: Mutex<Option<TransactionState>>,
    /// Live multi-batch indices keyed by table name.
    /// Rebuilt after every DML that touches the table.
    indices:     RwLock<HashMap<String, crate::storage::index::MultiTableRegistry>>,
}

impl Database {
    /// Create an in-memory database with no WAL.
    pub fn in_memory() -> Self {
        Self {
            tables:      RwLock::new(HashMap::new()),
            wal:         None,
            plan_cache:  RwLock::new(HashMap::new()),
            replaying:   AtomicBool::new(false),
            transaction: Mutex::new(None),
            indices:     RwLock::new(HashMap::new()),
        }
    }

    /// Create a database backed by a WAL file.
    ///
    /// Opening does not replay anything — load the base tables first, then call
    /// [`Database::recover`].
    pub fn with_wal(wal_path: impl AsRef<Path>) -> FalconResult<Self> {
        let wal = Wal::open(wal_path)?;
        Ok(Self {
            tables:      RwLock::new(HashMap::new()),
            wal:         Some(RwLock::new(wal)),
            plan_cache:  RwLock::new(HashMap::new()),
            replaying:   AtomicBool::new(false),
            transaction: Mutex::new(None),
            indices:     RwLock::new(HashMap::new()),
        })
    }

    // ─── Data loading ─────────────────────────────────────────────────────────

    pub fn load_csv(&self, path: impl AsRef<Path>, table_name: &str) -> FalconResult<usize> {
        let table = read_csv(path, table_name)?;
        let rows  = table.row_count();
        self.tables.write().unwrap()
            .insert(table_name.to_string(), Arc::new(table));
        Ok(rows)
    }

    pub fn load_csv_with_options(
        &self,
        path:       impl AsRef<Path>,
        table_name: &str,
        opts:       CsvReadOptions,
    ) -> FalconResult<usize> {
        let table = read_csv_with_options(path, table_name, opts)?;
        let rows  = table.row_count();
        self.tables.write().unwrap()
            .insert(table_name.to_string(), Arc::new(table));
        Ok(rows)
    }

    pub fn load_fdb(&self, path: impl AsRef<Path>, table_name: &str) -> FalconResult<usize> {
        let table = read_table(path, table_name)?;
        let rows  = table.row_count();
        self.tables.write().unwrap()
            .insert(table_name.to_string(), Arc::new(table));
        Ok(rows)
    }

    pub fn save_fdb(&self, table_name: &str, path: impl AsRef<Path>) -> FalconResult<()> {
        let tables = self.tables.read().unwrap();
        let table  = tables.get(table_name)
            .ok_or_else(|| FalconError::InvalidQuery(
                format!("table '{table_name}' not found")
            ))?;
        write_table(table, path)
    }

    /// Write a table to a Parquet file (Snappy-compressed).
    pub fn save_parquet(&self, table_name: &str, path: impl AsRef<Path>) -> FalconResult<()> {
        let tables = self.tables.read().unwrap();
        let table  = tables.get(table_name)
            .ok_or_else(|| FalconError::InvalidQuery(
                format!("table '{table_name}' not found")
            ))?;
        crate::storage::parquet_io::write_parquet(&table.batches, path.as_ref())
    }

    /// Load a Parquet file into a new (or existing) table.
    /// Register externally-produced batches as a table, replacing any table of
    /// that name.
    ///
    /// This is the entry point for data that did not come from a file FalconDB
    /// read itself — Arrow IPC, an FFI caller, a computed result set. The schema
    /// is taken from the first batch, and `append_batch` rejects any later batch
    /// that disagrees with it.
    pub fn create_table_from_batches(
        &self,
        table_name: &str,
        batches:    Vec<RecordBatch>,
    ) -> FalconResult<usize> {
        let schema = batches
            .first()
            .map(|b| b.schema.clone())
            .ok_or_else(|| {
                FalconError::InvalidQuery(
                    "cannot create a table from an empty batch list: no schema to infer".into(),
                )
            })?;

        let mut table = Table::new(table_name, schema);
        for batch in batches {
            table.append_batch(batch)?;
        }
        let rows = table.row_count();

        self.tables.write().unwrap().insert(table_name.to_string(), Arc::new(table));
        // Any index built over a previous table of this name is now stale.
        self.indices.write().unwrap().remove(table_name);
        Ok(rows)
    }

    pub fn load_parquet(&self, path: impl AsRef<Path>, table_name: &str) -> FalconResult<usize> {
        let batches = crate::storage::parquet_io::read_parquet(path.as_ref(), table_name)?;
        let total: usize = batches.iter().map(|b| b.row_count()).sum();
        let mut tables = self.tables.write().unwrap();
        let entry = tables.entry(table_name.to_string());
        match entry {
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let tbl = Arc::get_mut(o.get_mut())
                    .ok_or_else(|| FalconError::InvalidQuery("table is borrowed".into()))?;
                for b in batches { tbl.append_batch(b)?; }
            }
            std::collections::hash_map::Entry::Vacant(v) => {
                // The schema comes from the first batch, but every batch has to be
                // appended. Keeping only the first silently truncated the table to
                // one Parquet batch — about 1024 rows — while still reporting the
                // file's full row count, so the loss looked like a successful load.
                let mut it = batches.into_iter();
                let first = it.next()
                    .ok_or_else(|| FalconError::InvalidQuery("empty parquet file".into()))?;
                let schema = first.schema.clone();
                let mut tbl = crate::storage::table::Table::new(table_name, schema);
                tbl.append_batch(first)?;
                for b in it {
                    tbl.append_batch(b)?;
                }
                v.insert(Arc::new(tbl));
            }
        }
        Ok(total)
    }

    // ─── Query execution ─────────────────────────────────────────────────────

    /// Execute any statement. `SELECT` returns its rows; all other statements
    /// return a single `rows_affected` column so every caller — REST, FFI, tests —
    /// goes through one entry point.
    pub fn sql(&self, query: &str) -> FalconResult<QueryResult> {
        // Only SELECT is cached: DML and DDL are executed once and must not be cached.
        {
            let cache = self.plan_cache.read().unwrap();
            if !cache.contains_key(query) {
                drop(cache);
                use crate::query::sql::{parse_statement, ParsedStatement};
                match parse_statement(query)? {
                    ParsedStatement::Query(plan) => {
                        self.plan_cache.write().unwrap()
                            .insert(query.to_string(), Arc::new(plan));
                    }
                    ParsedStatement::Insert(stmt) => {
                        let r = self.run_insert(&stmt)?;
                        self.journal(WalEntryType::Insert, &stmt.table, query)?;
                        return Ok(r);
                    }
                    ParsedStatement::Update(stmt) => {
                        let r = self.run_update(&stmt)?;
                        self.journal(WalEntryType::Update, &stmt.table, query)?;
                        return Ok(r);
                    }
                    ParsedStatement::Delete(stmt) => {
                        let r = self.run_delete(&stmt)?;
                        self.journal(WalEntryType::Delete, &stmt.table, query)?;
                        return Ok(r);
                    }
                    ParsedStatement::CreateTable(stmt) => {
                        return Ok(self.run_create_table(stmt)?);
                    }
                    ParsedStatement::DropTable { name, if_exists } => {
                        return Ok(self.run_drop_table(&name, if_exists)?);
                    }
                    ParsedStatement::AlterTable(stmt) => {
                        return Ok(self.run_alter_table(stmt)?);
                    }
                    ParsedStatement::Explain { stmt, analyze } => {
                        return if analyze {
                            self.run_explain_analyze(*stmt, query)
                        } else {
                            Ok(run_explain(*stmt))
                        };
                    }
                    ParsedStatement::Truncate { name } => {
                        return self.run_truncate(&name);
                    }
                    ParsedStatement::Vacuum { table } => {
                        return self.run_vacuum(table.as_deref());
                    }
                    ParsedStatement::Copy { table, path, format } => {
                        return self.run_copy(&table, &path, format);
                    }
                    ParsedStatement::Checkpoint { path } => {
                        return self.run_checkpoint(path.as_deref());
                    }
                    ParsedStatement::Begin => {
                        return self.begin_transaction();
                    }
                    ParsedStatement::Commit => {
                        return self.commit_transaction();
                    }
                    ParsedStatement::Rollback => {
                        return self.rollback_transaction();
                    }
                    ParsedStatement::CreateIndex(stmt) => {
                        return self.run_create_index(stmt);
                    }
                    ParsedStatement::DropIndex { name, if_exists } => {
                        return self.run_drop_index(&name, if_exists);
                    }
                }
            }
        }
        self.run_select(query)
    }

    // ─── DML ──────────────────────────────────────────────────────────────────
    //
    // All three mutations hold the write lock for the entire operation — read,
    // compute new version, and install — so concurrent mutations see each other's
    // changes and never silently clobber a concurrent write.

    fn run_insert(&self, stmt: &crate::query::dml::InsertStmt) -> FalconResult<QueryResult> {
        use crate::query::dml::InsertSource;

        // For INSERT … SELECT, execute the sub-query under a READ lock before
        // acquiring the WRITE lock (RwLock is exclusive for writers).
        let resolved_stmt;
        let stmt = if let InsertSource::Select(query) = &stmt.source {
            let batches = {
                let tables  = self.tables.read().unwrap();
                let indices = self.indices.read().unwrap();
                execute_plan(query, &tables, &HashMap::new(), &indices)?
            };
            // Materialise every row as Literal expressions so execute_insert sees Values.
            let rows: Vec<Vec<crate::query::expr::Expr>> = batches.iter()
                .flat_map(|batch| {
                    (0..batch.row_count()).map(move |r| {
                        (0..batch.column_count())
                            .map(|c| {
                                batch.column(c).get(r)
                                    .map(crate::query::expr::Expr::Literal)
                            })
                            .collect::<FalconResult<Vec<_>>>()
                    })
                })
                .collect::<FalconResult<_>>()?;
            resolved_stmt = crate::query::dml::InsertStmt {
                table:     stmt.table.clone(),
                columns:   stmt.columns.clone(),
                source:    InsertSource::Values(rows),
                returning: stmt.returning.clone(),
            };
            &resolved_stmt
        } else {
            stmt
        };

        let mut tables = self.tables.write().unwrap();
        let (updated, n, ret) = {
            let table = tables.get(&stmt.table).ok_or_else(|| {
                FalconError::InvalidQuery(format!("table '{}' not found", stmt.table))
            })?;
            crate::query::dml::execute_insert(table, stmt)?
        };
        enforce_constraints(&updated)?;
        enforce_foreign_keys(&updated, &tables)?;
        let table_name = stmt.table.clone();
        tables.insert(table_name.clone(), Arc::new(updated));
        drop(tables);
        self.rebuild_table_indices(&table_name)?;
        Ok(match ret {
            Some(batch) => QueryResult { batches: vec![batch] },
            None        => rows_affected(n),
        })
    }

    fn run_update(&self, stmt: &crate::query::dml::UpdateStmt) -> FalconResult<QueryResult> {
        let (table_name, n, ret) = {
            let mut tables = self.tables.write().unwrap();
            let (updated, n, ret) = {
                let table = tables.get(&stmt.table).ok_or_else(|| {
                    FalconError::InvalidQuery(format!("table '{}' not found", stmt.table))
                })?;
                crate::query::dml::execute_update(table, stmt)?
            };
            enforce_constraints(&updated)?;
            enforce_foreign_keys(&updated, &tables)?;
            tables.insert(stmt.table.clone(), Arc::new(updated));
            (stmt.table.clone(), n, ret)
        };
        self.rebuild_table_indices(&table_name)?;
        Ok(match ret {
            Some(batch) => QueryResult { batches: vec![batch] },
            None        => rows_affected(n),
        })
    }

    fn run_delete(&self, stmt: &crate::query::dml::DeleteStmt) -> FalconResult<QueryResult> {
        let (table_name, n, ret) = {
            let mut tables = self.tables.write().unwrap();
            let (tbl, n, ret) = {
                let table = tables.get(&stmt.table).ok_or_else(|| {
                    FalconError::InvalidQuery(format!("table '{}' not found", stmt.table))
                })?;
                crate::query::dml::execute_delete(table, stmt)?
            };
            tables.insert(stmt.table.clone(), Arc::new(tbl));
            (stmt.table.clone(), n, ret)
        };
        self.rebuild_table_indices(&table_name)?;
        Ok(match ret {
            Some(batch) => QueryResult { batches: vec![batch] },
            None        => rows_affected(n),
        })
    }

    // ─── Durability ───────────────────────────────────────────────────────────

    /// Record a mutation in the write-ahead log.
    ///
    /// Inside a transaction, entries are buffered and flushed atomically on COMMIT.
    /// Outside a transaction, they are written immediately.
    ///
    /// Called after the in-memory mutation succeeds, so a statement that fails
    /// validation never reaches the log.
    fn journal(&self, kind: WalEntryType, table: &str, sql: &str) -> FalconResult<()> {
        if self.replaying.load(Ordering::Relaxed) {
            return Ok(());
        }
        // Substitute NOW()/CURRENT_TIMESTAMP with literal values so replaying the
        // WAL is deterministic regardless of when the replay happens.
        let frozen = freeze_now_calls(sql);
        let sql = frozen.as_deref().unwrap_or(sql);

        // Inside a transaction: buffer instead of writing immediately.
        {
            let mut txn = self.transaction.lock().unwrap();
            if let Some(state) = txn.as_mut() {
                state.wal_buffer.push((kind, table.to_string(), sql.to_string()));
                return Ok(());
            }
        }
        self.write_wal(kind, table, sql)
    }

    fn write_wal(&self, kind: WalEntryType, table: &str, sql: &str) -> FalconResult<()> {
        let Some(wal) = &self.wal else { return Ok(()) };
        let mut wal = wal.write().unwrap();
        let payload = sql.as_bytes();
        match kind {
            WalEntryType::Insert => wal.write_insert(table, payload)?,
            WalEntryType::Update => wal.write_update(table, payload)?,
            WalEntryType::Delete => wal.write_delete(table, payload)?,
            other => {
                return Err(FalconError::InvalidQuery(format!(
                    "cannot journal entry type {other:?}"
                )))
            }
        };
        Ok(())
    }

    // ─── Transaction management ───────────────────────────────────────────────

    fn begin_transaction(&self) -> FalconResult<QueryResult> {
        // Check without holding the tables lock to avoid ABBA deadlock:
        // DML path is tables.write → transaction.lock; so here we must not
        // hold transaction.lock while taking tables.read.
        {
            let txn = self.transaction.lock().unwrap();
            if txn.is_some() {
                return Err(FalconError::InvalidQuery(
                    "cannot BEGIN: already inside a transaction".into(),
                ));
            }
        }
        let snapshot = self.tables.read().unwrap().clone();
        {
            let mut txn = self.transaction.lock().unwrap();
            if txn.is_some() {
                return Err(FalconError::InvalidQuery(
                    "cannot BEGIN: already inside a transaction".into(),
                ));
            }
            *txn = Some(TransactionState { snapshot, wal_buffer: Vec::new() });
        }
        Ok(rows_affected(0))
    }

    fn commit_transaction(&self) -> FalconResult<QueryResult> {
        let state = {
            let mut txn = self.transaction.lock().unwrap();
            txn.take().ok_or_else(|| {
                FalconError::InvalidQuery("COMMIT outside of a transaction".into())
            })?
        };
        // Flush buffered WAL entries (WAL lock taken after transaction lock released).
        for (kind, table, sql) in state.wal_buffer {
            self.write_wal(kind, &table, &sql)?;
        }
        Ok(rows_affected(0))
    }

    fn rollback_transaction(&self) -> FalconResult<QueryResult> {
        let state = {
            let mut txn = self.transaction.lock().unwrap();
            txn.take().ok_or_else(|| {
                FalconError::InvalidQuery("ROLLBACK outside of a transaction".into())
            })?
        };
        // Restore snapshot (tables lock taken after transaction lock released).
        *self.tables.write().unwrap() = state.snapshot;
        // Invalidate the plan cache: DDL inside the transaction may have changed schemas.
        self.plan_cache.write().unwrap().clear();
        Ok(rows_affected(0))
    }

    /// Re-apply logged mutations on top of the currently loaded tables.
    ///
    /// Load the base data first (`load_csv` / `load_fdb`), then call this. Returns
    /// the number of statements replayed. Entries for tables that are not loaded
    /// are skipped rather than treated as an error, so a WAL covering several
    /// tables can be replayed against a partial working set.
    pub fn recover(&self) -> FalconResult<usize> {
        let Some(wal) = &self.wal else { return Ok(0) };

        let entries = {
            let wal = wal.read().unwrap();
            crate::storage::wal::read_wal(wal.path())?
        };

        self.replaying.store(true, Ordering::Relaxed);
        let result = (|| {
            let mut replayed = 0usize;
            for entry in &entries {
                if !matches!(
                    entry.kind,
                    WalEntryType::Insert | WalEntryType::Update | WalEntryType::Delete
                ) {
                    continue;
                }
                if !self.tables.read().unwrap().contains_key(&entry.table) {
                    continue;
                }
                let sql = std::str::from_utf8(&entry.payload).map_err(|_| {
                    FalconError::ParseError(format!(
                        "WAL entry {} payload is not valid UTF-8",
                        entry.seq
                    ))
                })?;
                self.sql(sql)?;
                replayed += 1;
            }
            Ok(replayed)
        })();
        self.replaying.store(false, Ordering::Relaxed);
        result
    }

    // ─── DDL ──────────────────────────────────────────────────────────────────

    fn run_create_table(
        &self,
        stmt: crate::query::sql::CreateTableStmt,
    ) -> FalconResult<QueryResult> {
        use crate::storage::{
            schema::{Field, Schema},
            table::{ConstraintKind, Table},
        };

        let mut tables = self.tables.write().unwrap();
        if tables.contains_key(&stmt.name) {
            if stmt.if_not_exists {
                return Ok(rows_affected(0));
            }
            return Err(FalconError::InvalidQuery(
                format!("table '{}' already exists", stmt.name)
            ));
        }

        // For table-level PRIMARY KEY constraints, mark referenced columns NOT NULL.
        let mut nullable_overrides: std::collections::HashMap<String, bool> =
            std::collections::HashMap::new();
        for tc in &stmt.constraints {
            if let ConstraintKind::PrimaryKey(cols) = &tc.kind {
                for col in cols {
                    nullable_overrides.insert(col.to_lowercase(), false);
                }
            }
        }

        let fields: Vec<Field> = stmt.columns.iter()
            .map(|c| {
                let nullable = *nullable_overrides
                    .get(&c.name.to_lowercase())
                    .unwrap_or(&c.nullable);
                Field::new(c.name.clone(), c.data_type.clone(), nullable)
            })
            .collect();
        let schema = Schema::new(fields);
        tables.insert(
            stmt.name.clone(),
            Arc::new(Table::with_constraints(&stmt.name, schema, stmt.constraints)),
        );
        Ok(rows_affected(0))
    }

    fn run_drop_table(&self, name: &str, if_exists: bool) -> FalconResult<QueryResult> {
        let mut tables = self.tables.write().unwrap();
        if tables.remove(name).is_none() && !if_exists {
            return Err(FalconError::InvalidQuery(
                format!("table '{name}' does not exist")
            ));
        }
        Ok(rows_affected(0))
    }

    fn run_alter_table(
        &self,
        stmt: crate::query::sql::AlterTableStmt,
    ) -> FalconResult<QueryResult> {
        use crate::query::sql::AlterAction;
        use crate::storage::{
            column::Column,
            schema::{Field, Schema},
            table::{RecordBatch, Table},
        };

        let mut tables = self.tables.write().unwrap();
        let table = tables.get(&stmt.table).ok_or_else(|| {
            FalconError::InvalidQuery(format!("table '{}' not found", stmt.table))
        })?;

        match &stmt.action {
            AlterAction::AddColumn(col_def) => {
                if table.schema.has_column(&col_def.name) {
                    return Err(FalconError::InvalidQuery(
                        format!("column '{}' already exists in '{}'", col_def.name, stmt.table)
                    ));
                }
                let mut new_fields = table.schema.fields.clone();
                new_fields.push(Field::new(
                    col_def.name.clone(),
                    col_def.data_type.clone(),
                    col_def.nullable,
                ));
                let new_schema = Schema::new(new_fields);

                let mut new_table = Table::with_constraints(
                    &stmt.table, new_schema.clone(), table.constraints.clone()
                );
                for batch in &table.batches {
                    let n = batch.row_count();
                    let null_col = Arc::new(make_null_column(&col_def.data_type, n));
                    let mut cols: Vec<Arc<Column>> = batch.columns.clone();
                    cols.push(null_col);
                    let new_batch = RecordBatch::from_shared(new_schema.clone(), cols)?;
                    new_table.append_batch(new_batch)?;
                }
                tables.insert(stmt.table.clone(), Arc::new(new_table));
            }

            AlterAction::DropColumn(col_name) => {
                let drop_idx = table.schema.column_index(col_name)?;
                let new_fields: Vec<Field> = table.schema.fields.iter()
                    .enumerate()
                    .filter(|(i, _)| *i != drop_idx)
                    .map(|(_, f)| f.clone())
                    .collect();
                let new_schema = Schema::new(new_fields);

                let mut new_table = Table::new(&stmt.table, new_schema.clone());
                for batch in &table.batches {
                    let cols: Vec<Arc<Column>> = batch.columns.iter()
                        .enumerate()
                        .filter(|(i, _)| *i != drop_idx)
                        .map(|(_, c)| Arc::clone(c))
                        .collect();
                    let new_batch = RecordBatch::from_shared(new_schema.clone(), cols)?;
                    new_table.append_batch(new_batch)?;
                }
                tables.insert(stmt.table.clone(), Arc::new(new_table));
            }

            AlterAction::AddConstraint(tc) => {
                let mut new_table = crate::query::dml::clone_table_arc(&table);
                new_table.constraints.push(tc.clone());
                // Validate existing data against the new constraint immediately.
                enforce_constraints(&new_table)?;
                enforce_foreign_keys(&new_table, &tables)?;
                tables.insert(stmt.table.clone(), Arc::new(new_table));
            }

            AlterAction::DropConstraint(name) => {
                let mut new_table = crate::query::dml::clone_table_arc(&table);
                let before = new_table.constraints.len();
                new_table.constraints.retain(|c| {
                    c.name.as_deref().map(|n| !n.eq_ignore_ascii_case(name)).unwrap_or(true)
                });
                if new_table.constraints.len() == before {
                    return Err(FalconError::InvalidQuery(
                        format!("constraint '{name}' not found on table '{}'", stmt.table)
                    ));
                }
                tables.insert(stmt.table.clone(), Arc::new(new_table));
            }
        }
        Ok(rows_affected(0))
    }

    fn run_truncate(&self, name: &str) -> FalconResult<QueryResult> {
        let mut tables = self.tables.write().unwrap();
        let table = tables.get(name).ok_or_else(|| {
            FalconError::InvalidQuery(format!("table '{name}' not found"))
        })?;
        let empty = crate::storage::table::Table {
            name:        table.name.clone(),
            schema:      table.schema.clone(),
            batches:     Vec::new(),
            stats:       Vec::new(),
            constraints: table.constraints.clone(),
            indices:     table.indices.clone(),
        };
        tables.insert(name.to_string(), Arc::new(empty));
        drop(tables);
        // After truncate, clear any live index for this table.
        self.indices.write().unwrap().remove(name);
        Ok(rows_affected(0))
    }

    /// `VACUUM [table]` — compact fragmented batches, returning the number of
    /// batches reclaimed.
    ///
    /// Only the batch layout changes; rows, values and ordering are untouched, so
    /// this is safe to run at any time. Tables are rebuilt one at a time rather
    /// than under a single long lock, since vacuuming copies data.
    fn run_vacuum(&self, table: Option<&str>) -> FalconResult<QueryResult> {
        let names: Vec<String> = match table {
            Some(n) => {
                if !self.tables.read().unwrap().contains_key(n) {
                    return Err(FalconError::InvalidQuery(format!("table '{n}' not found")));
                }
                vec![n.to_string()]
            }
            None => self.tables.read().unwrap().keys().cloned().collect(),
        };

        let mut reclaimed = 0usize;
        for name in names {
            // Take a handle out of the map so the rebuild happens off-lock.
            let Some(existing) = self.tables.read().unwrap().get(&name).cloned() else {
                continue; // dropped concurrently; nothing to compact
            };
            // Batches hold their columns behind Arc, so this copies pointers, not data.
            let mut owned = crate::storage::table::Table {
                name:        existing.name.clone(),
                schema:      existing.schema.clone(),
                batches:     existing.batches.clone(),
                stats:       existing.stats.clone(),
                constraints: existing.constraints.clone(),
                indices:     existing.indices.clone(),
            };
            reclaimed += owned.vacuum()?;
            self.tables.write().unwrap().insert(name, Arc::new(owned));
        }

        Ok(rows_affected(reclaimed))
    }

    /// `COPY <table> FROM '<path>'` — bulk-load a file, replacing the table.
    ///
    /// Returns the number of rows loaded. Replacement rather than append matches
    /// the existing `load_csv`/`load_parquet` entry points, so the SQL surface and
    /// the Rust API agree on what loading a file means.
    fn run_copy(
        &self,
        table:  &str,
        path:   &str,
        format: crate::query::sql::CopyFormat,
    ) -> FalconResult<QueryResult> {
        use crate::query::sql::CopyFormat;

        let p = std::path::Path::new(path);
        let is_glob = path.contains('*') || path.contains('?');
        if !is_glob && !p.exists() {
            return Err(FalconError::InvalidQuery(format!("no such file: {path}")));
        }

        // `COPY t FROM 'data/*.parquet'` loads every matching file as one table.
        if is_glob {
            if format != CopyFormat::Parquet {
                return Err(FalconError::InvalidQuery(
                    "glob patterns are only supported for Parquet".into(),
                ));
            }
            let batches = crate::storage::parquet_io::read_parquet_glob(path, table)?;
            let rows = self.create_table_from_batches(table, batches)?;
            return Ok(rows_affected(rows));
        }

        let rows = match format {
            CopyFormat::Csv     => self.load_csv(p, table)?,
            CopyFormat::Parquet => {
                // load_parquet appends to an existing table, so drop it first to
                // keep COPY's replace semantics.
                self.tables.write().unwrap().remove(table);
                self.load_parquet(p, table)?
            }
        };
        // Any index built over the old contents no longer describes this table.
        self.indices.write().unwrap().remove(table);
        Ok(rows_affected(rows))
    }

    // ─── Index DDL ────────────────────────────────────────────────────────────

    fn run_create_index(
        &self,
        stmt: crate::query::sql::CreateIndexStmt,
    ) -> FalconResult<QueryResult> {
        use crate::storage::{
            index::{MultiTableRegistry, TableIndex},
            table::IndexDef,
        };

        let mut tables = self.tables.write().unwrap();
        let table = tables.get(&stmt.table).ok_or_else(|| {
            FalconError::InvalidQuery(format!("table '{}' not found", stmt.table))
        })?;

        // Check for duplicate index name on this table.
        if table.indices.iter().any(|d| d.name.eq_ignore_ascii_case(&stmt.name)) {
            if stmt.if_not_exists {
                return Ok(rows_affected(0));
            }
            return Err(FalconError::InvalidQuery(
                format!("index '{}' already exists on '{}'", stmt.name, stmt.table)
            ));
        }

        // Build the live index.
        let idx = match &stmt.index_type {
            crate::storage::table::IndexType::Hash =>
                TableIndex::build_hash(&stmt.column, table)?,
            crate::storage::table::IndexType::BTree =>
                TableIndex::build_btree(&stmt.column, table)?,
        };

        // Store the index metadata in the table.
        let mut new_table = crate::query::dml::clone_table_arc(table);
        new_table.indices.push(IndexDef {
            name:       stmt.name.clone(),
            column:     stmt.column.clone(),
            index_type: stmt.index_type,
            unique:     stmt.unique,
        });
        tables.insert(stmt.table.clone(), Arc::new(new_table));
        drop(tables);

        // Store the live index in the registry.
        let mut indices = self.indices.write().unwrap();
        let reg = indices.entry(stmt.table.clone()).or_insert_with(MultiTableRegistry::new);
        reg.add(idx);

        Ok(rows_affected(0))
    }

    fn run_drop_index(&self, name: &str, if_exists: bool) -> FalconResult<QueryResult> {
        let mut tables = self.tables.write().unwrap();

        // Find which table owns this index.
        let table_name = tables.iter()
            .find(|(_, t)| t.indices.iter().any(|d| d.name.eq_ignore_ascii_case(name)))
            .map(|(n, _)| n.clone());

        let Some(table_name) = table_name else {
            if if_exists {
                return Ok(rows_affected(0));
            }
            return Err(FalconError::InvalidQuery(
                format!("index '{name}' does not exist")
            ));
        };

        // Remove the IndexDef from the table.
        let table = tables.get(&table_name).unwrap();
        let col_name = table.indices.iter()
            .find(|d| d.name.eq_ignore_ascii_case(name))
            .map(|d| d.column.clone())
            .unwrap();

        let mut new_table = crate::query::dml::clone_table_arc(table);
        new_table.indices.retain(|d| !d.name.eq_ignore_ascii_case(name));
        tables.insert(table_name.clone(), Arc::new(new_table));
        drop(tables);

        // Remove the live index.
        let mut indices = self.indices.write().unwrap();
        if let Some(reg) = indices.get_mut(&table_name) {
            reg.remove(&col_name);
        }

        Ok(rows_affected(0))
    }

    /// Rebuild all live indices for a table after DML.
    fn rebuild_table_indices(&self, table_name: &str) -> FalconResult<()> {
        use crate::storage::index::{MultiTableRegistry, TableIndex};

        let index_defs: Vec<crate::storage::table::IndexDef> = {
            let tables = self.tables.read().unwrap();
            tables.get(table_name)
                .map(|t| t.indices.clone())
                .unwrap_or_default()
        };

        if index_defs.is_empty() {
            return Ok(());
        }

        let tables = self.tables.read().unwrap();
        let table = tables.get(table_name).unwrap();
        let mut reg = MultiTableRegistry::new();
        for def in &index_defs {
            let idx = match &def.index_type {
                crate::storage::table::IndexType::Hash =>
                    TableIndex::build_hash(&def.column, table)?,
                crate::storage::table::IndexType::BTree =>
                    TableIndex::build_btree(&def.column, table)?,
            };
            reg.add(idx);
        }
        drop(tables);

        self.indices.write().unwrap().insert(table_name.to_string(), reg);
        Ok(())
    }

    fn run_checkpoint(&self, explicit_path: Option<&str>) -> FalconResult<QueryResult> {
        let base_dir: std::path::PathBuf = match explicit_path {
            Some(p) => std::path::PathBuf::from(p),
            None => {
                // Derive from WAL path; fail gracefully if there is no WAL.
                if let Some(wal) = &self.wal {
                    let wal = wal.read().unwrap();
                    wal.path()
                        .parent()
                        .ok_or_else(|| FalconError::InvalidQuery(
                            "cannot derive checkpoint directory from WAL path".into()
                        ))?
                        .to_path_buf()
                } else {
                    return Err(FalconError::InvalidQuery(
                        "CHECKPOINT requires a path for in-memory databases".into()
                    ));
                }
            }
        };
        let n = self.checkpoint(&base_dir)?;
        Ok(rows_affected(n))
    }

    /// Save every table to `<base_dir>/<name>.fdb` and truncate the WAL.
    ///
    /// After a checkpoint the WAL file is empty: recovery only needs to load the
    /// `.fdb` files.  Tables that were not created with a WAL-backed `Database`
    /// are written to disk but the truncation step is skipped (no-op).
    pub fn checkpoint(&self, base_dir: &std::path::Path) -> FalconResult<usize> {
        std::fs::create_dir_all(base_dir)?;

        let tables = self.tables.read().unwrap();
        let mut count = 0usize;
        for (name, table) in tables.iter() {
            let path = base_dir.join(format!("{name}.fdb"));
            write_table(table, path)?;
            count += 1;
        }
        drop(tables);

        if let Some(wal) = &self.wal {
            wal.write().unwrap().truncate()?;
        }

        Ok(count)
    }

    fn run_select(&self, query: &str) -> FalconResult<QueryResult> {
        let parsed: Arc<crate::query::sql::ParsedQuery> = {
            let cache = self.plan_cache.read().unwrap();
            match cache.get(query) {
                Some(plan) => Arc::clone(plan),
                None => {
                    drop(cache);
                    let plan = Arc::new(parse_sql(query)?);
                    self.plan_cache.write().unwrap()
                        .insert(query.to_string(), Arc::clone(&plan));
                    plan
                }
            }
        };
        let tables  = self.tables.read().unwrap();
        let indices = self.indices.read().unwrap();
        let batches = execute_plan(&parsed, &tables, &HashMap::new(), &indices)?;
        Ok(QueryResult { batches })
    }
}

// ─── Constraint enforcement ───────────────────────────────────────────────────

/// Verify all integrity constraints on `table` after a write.
///
/// Scans the entire table (all batches) for:
///   1. NOT NULL  — fields with `nullable = false` must have no null values.
///   2. PRIMARY KEY / UNIQUE  — combined key tuples must be unique across all rows.
///      NULLs are exempt from UNIQUE uniqueness (SQL standard); PRIMARY KEY NULLs
///      are caught by the NOT NULL check first.
///   3. CHECK  — the boolean expression must not evaluate to FALSE for any row.
///      NULL evaluations (three-valued logic) are treated as "passes".
fn enforce_constraints(table: &crate::storage::table::Table) -> FalconResult<()> {
    use std::collections::HashSet;
    use crate::storage::{
        table::ConstraintKind,
        types::{FalconError, ScalarValue},
    };

    // ── 1. NOT NULL ───────────────────────────────────────────────────────────
    for (col_idx, field) in table.schema.fields.iter().enumerate() {
        if !field.nullable {
            for batch in &table.batches {
                if batch.column(col_idx).null_count() > 0 {
                    return Err(FalconError::ConstraintViolation(format!(
                        "NOT NULL violated on column '{}'", field.name
                    )));
                }
            }
        }
    }

    // ── 2 & 3. Table constraints ──────────────────────────────────────────────
    for tc in &table.constraints {
        match &tc.kind {
            // ── PRIMARY KEY / UNIQUE ──────────────────────────────────────────
            ConstraintKind::PrimaryKey(cols) | ConstraintKind::Unique(cols) => {
                let is_pk = matches!(&tc.kind, ConstraintKind::PrimaryKey(_));
                let col_indices: Vec<usize> = cols.iter()
                    .map(|c| table.schema.column_index(c))
                    .collect::<FalconResult<_>>()?;

                let mut seen: HashSet<Vec<String>> = HashSet::new();
                for batch in &table.batches {
                    for row in 0..batch.row_count() {
                        let key: Vec<String> = col_indices.iter()
                            .map(|&ci| batch.column(ci).get(row).map(|v| v.to_string()))
                            .collect::<FalconResult<_>>()?;

                        let has_null = key.iter().any(|v| {
                            v == "NULL" || v == "null"
                        });

                        // For UNIQUE: NULLs never violate uniqueness.
                        // For PRIMARY KEY: NULLs are already caught by NOT NULL above.
                        if has_null && !is_pk {
                            continue;
                        }

                        if !seen.insert(key.clone()) {
                            let kind = if is_pk { "PRIMARY KEY" } else { "UNIQUE" };
                            return Err(FalconError::ConstraintViolation(format!(
                                "{kind} constraint violated: duplicate key ({})",
                                key.join(", ")
                            )));
                        }
                    }
                }
            }

            // ── CHECK ─────────────────────────────────────────────────────────
            ConstraintKind::Check(sql_expr) => {
                let expr = crate::query::sql::parse_check_expr(sql_expr)?;
                for batch in &table.batches {
                    let values = expr.eval_batch(batch)?;
                    for (row, val) in values.iter().enumerate() {
                        match val {
                            ScalarValue::Boolean(Some(false)) => {
                                let name = tc.name.as_deref().unwrap_or(sql_expr.as_str());
                                return Err(FalconError::ConstraintViolation(format!(
                                    "CHECK constraint '{name}' violated at row {row}"
                                )));
                            }
                            // true or NULL → passes (SQL three-valued logic)
                            _ => {}
                        }
                    }
                }
            }

            // FK enforcement requires the referenced table — handled separately in
            // enforce_foreign_keys(), which receives the full tables map.
            ConstraintKind::ForeignKey { .. } => {}
        }
    }
    Ok(())
}

/// Verify all FOREIGN KEY constraints on `table`, looking up referenced tables in `tables`.
/// NULLs in the FK columns are exempt (SQL standard — partial NULL keys do not violate FK).
fn enforce_foreign_keys(
    table:  &crate::storage::table::Table,
    tables: &HashMap<String, Arc<crate::storage::table::Table>>,
) -> FalconResult<()> {
    use std::collections::HashSet;
    use crate::storage::table::ConstraintKind;
    use crate::storage::types::FalconError;

    for tc in &table.constraints {
        let ConstraintKind::ForeignKey { columns, ref_table, ref_columns } = &tc.kind else {
            continue;
        };

        let ref_tbl = tables.get(ref_table).ok_or_else(|| {
            FalconError::ConstraintViolation(format!(
                "FOREIGN KEY references unknown table '{ref_table}'"
            ))
        })?;

        // Build the set of valid (ref_col) tuples from the referenced table.
        let ref_indices: Vec<usize> = ref_columns.iter()
            .map(|c| ref_tbl.schema.column_index(c))
            .collect::<FalconResult<_>>()?;

        let mut valid: HashSet<Vec<String>> = HashSet::new();
        for batch in &ref_tbl.batches {
            for row in 0..batch.row_count() {
                let key: Vec<String> = ref_indices.iter()
                    .map(|&ci| batch.column(ci).get(row).map(|v| v.to_string()))
                    .collect::<FalconResult<_>>()?;
                valid.insert(key);
            }
        }

        // Check every row in the mutated table.
        let col_indices: Vec<usize> = columns.iter()
            .map(|c| table.schema.column_index(c))
            .collect::<FalconResult<_>>()?;

        for batch in &table.batches {
            for row in 0..batch.row_count() {
                let key: Vec<String> = col_indices.iter()
                    .map(|&ci| batch.column(ci).get(row).map(|v| v.to_string()))
                    .collect::<FalconResult<_>>()?;

                // Any NULL in the FK → row is exempt.
                let is_null = key.iter().any(|v| matches!(
                    v.as_str(), "NULL" | "null"
                ));
                if is_null {
                    continue;
                }

                if !valid.contains(&key) {
                    let cname = tc.name.as_deref()
                        .unwrap_or_else(|| columns.first().map(|s| s.as_str()).unwrap_or("?"));
                    return Err(FalconError::ConstraintViolation(format!(
                        "FOREIGN KEY constraint '{cname}' violated: ({}) not found in {ref_table}",
                        key.join(", ")
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Replace `NOW()`, `CURRENT_TIMESTAMP`, `CURRENT_DATE` with literal values so the
/// SQL logged to the WAL is deterministic on replay.
///
/// Returns `None` when the SQL contains none of these tokens (fast path).
fn freeze_now_calls(sql: &str) -> Option<String> {
    let upper = sql.to_ascii_uppercase();
    if !upper.contains("NOW()") &&
       !upper.contains("CURRENT_TIMESTAMP") &&
       !upper.contains("CURRENT_DATE") {
        return None;
    }

    use std::time::{SystemTime, UNIX_EPOCH};
    let micros = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);
    let day = micros.div_euclid(1_000_000 * 86_400) as i32;
    let ts_literal  = format!("TIMESTAMP '{micros}'");
    let date_literal = format!("DATE '{day}'");

    // Case-insensitive replacement, preserving rest of SQL.
    // We scan the original SQL and replace matches case-insensitively.
    let mut out = String::with_capacity(sql.len() + 32);
    let mut rest = sql;
    while !rest.is_empty() {
        let up = rest.to_ascii_uppercase();
        if up.starts_with("NOW()") {
            out.push_str(&ts_literal);
            rest = &rest["NOW()".len()..];
        } else if up.starts_with("CURRENT_TIMESTAMP") {
            out.push_str(&ts_literal);
            rest = &rest["CURRENT_TIMESTAMP".len()..];
        } else if up.starts_with("CURRENT_DATE") {
            out.push_str(&date_literal);
            rest = &rest["CURRENT_DATE".len()..];
        } else {
            // Advance by one char.
            let ch = rest.chars().next().unwrap();
            out.push(ch);
            rest = &rest[ch.len_utf8()..];
        }
    }
    Some(out)
}

/// Run one plan against a set of tables plus any in-scope CTE bindings.
///
/// Split out of `Database` so CTEs — which are just plans that must be evaluated
/// before the query that names them — can recurse through the same code path.
fn execute_plan(
    parsed:  &crate::query::sql::ParsedQuery,
    tables:  &HashMap<String, Arc<Table>>,
    ctes:    &HashMap<String, Arc<Table>>,
    indices: &HashMap<String, crate::storage::index::MultiTableRegistry>,
) -> FalconResult<Vec<RecordBatch>> {
    use crate::query::select::SelectQuery;

    // Materialise each CTE in declaration order so a later one can reference an
    // earlier one, then make them visible to this query only.
    let mut scope: HashMap<String, Arc<Table>> = ctes.clone();
    for (name, cte_plan, recursive) in &parsed.ctes {
        if !recursive {
            let cte_batches = execute_plan(cte_plan, tables, &scope, indices)?;
            scope.insert(name.clone(), Arc::new(table_from_batches(name, cte_batches)?));
        } else {
            // WITH RECURSIVE: iterative evaluation.
            // The CTE plan's set_ops should be [(UnionAll|Distinct, recursive_part)].
            // We run the anchor first, then iterate the recursive part until no new rows.
            // Only one recursive term is handled. Taking just the first set op and
            // ignoring the rest would silently drop query terms, so reject it.
            if cte_plan.set_ops.len() > 1 {
                return Err(FalconError::InvalidQuery(format!(
                    "recursive CTE '{name}' has {} set operations; only a single                      anchor UNION [ALL] recursive term is supported",
                    cte_plan.set_ops.len()
                )));
            }
            let (anchor_plan, rec_plan, is_union_all) = match cte_plan.set_ops.first() {
                Some((op, rec)) => {
                    use crate::query::sql::SetOp;
                    let is_union_all = match op {
                        SetOp::UnionAll      => true,
                        SetOp::UnionDistinct => false,
                        other => return Err(FalconError::InvalidQuery(format!(
                            "recursive CTE '{name}' must combine its terms with                              UNION or UNION ALL, not {other:?}"
                        ))),
                    };
                    let anchor = crate::query::sql::ParsedQuery {
                        set_ops: vec![],
                        ctes:    vec![],
                        ..cte_plan.clone()
                    };
                    (anchor, rec.clone(), is_union_all)
                }
                None => {
                    // No UNION — just a non-recursive CTE marked recursive (harmless)
                    let cte_batches = execute_plan(cte_plan, tables, &scope, indices)?;
                    scope.insert(name.clone(), Arc::new(table_from_batches(name, cte_batches)?));
                    continue;
                }
            };

            // 1. Execute anchor. Its column names are the CTE's column names.
            let anchor_batches = execute_plan(&anchor_plan, tables, &scope, indices)?;
            let cte_columns: Vec<String> = anchor_batches
                .first()
                .map(|b| b.schema.fields.iter().map(|f| f.name.clone()).collect())
                .unwrap_or_default();
            let mut all_batches: Vec<RecordBatch> = anchor_batches.clone();
            let mut prev_batches = anchor_batches;

            let mut converged = false;
            for _iter in 0..MAX_RECURSIVE_ITERS {
                if prev_batches.is_empty() || prev_batches.iter().all(|b| b.row_count() == 0) {
                    converged = true;
                    break;
                }
                // Expose the previous result as the CTE for this iteration.
                scope.insert(name.clone(), Arc::new(table_from_batches(name, prev_batches.clone())?));

                // The recursive term is matched to the anchor by position, so its
                // output is renamed to the anchor's column names before it is
                // accumulated or fed back in. Without this, an unaliased `n + 1`
                // would be exposed as a column called "n + 1" and the next pass
                // would not find `n`.
                let new_batches = rename_batches_positionally(
                    execute_plan(&rec_plan, tables, &scope, indices)?,
                    &cte_columns,
                )?;
                if new_batches.is_empty() || new_batches.iter().all(|b| b.row_count() == 0) {
                    converged = true;
                    break;
                }

                if !is_union_all {
                    use crate::storage::types::ScalarValue;
                    // UNION (distinct): filter out rows already in all_batches.
                    // Use string-key dedup — same approach as UNION DISTINCT in set ops.
                    let seen: std::collections::HashSet<String> = all_batches.iter()
                        .flat_map(|b| (0..b.row_count()).map(move |r| {
                            (0..b.schema.fields.len())
                                .map(|c| b.column(c).get(r).unwrap_or(ScalarValue::Null).to_string())
                                .collect::<Vec<_>>().join("\x00")
                        }))
                        .collect();
                    let fresh: Vec<RecordBatch> = new_batches.into_iter().filter_map(|b| {
                        let keep: Vec<bool> = (0..b.row_count()).map(|r| {
                            let key: String = (0..b.schema.fields.len())
                                .map(|c| b.column(c).get(r).unwrap_or(ScalarValue::Null).to_string())
                                .collect::<Vec<_>>().join("\x00");
                            !seen.contains(&key)
                        }).collect();
                        filter_batch_by_mask(&b, &keep)
                    }).collect();
                    if fresh.is_empty() || fresh.iter().all(|b| b.row_count() == 0) {
                        converged = true;
                        break;
                    }
                    all_batches.extend(fresh.clone());
                    prev_batches = fresh;
                } else {
                    all_batches.extend(new_batches.clone());
                    prev_batches = new_batches;
                }
            }

            if !converged {
                return Err(FalconError::InvalidQuery(format!(
                    "recursive CTE '{name}' did not reach a fixed point within                      {MAX_RECURSIVE_ITERS} iterations"
                )));
            }

            scope.insert(name.clone(), Arc::new(table_from_batches(name, all_batches)?));
        }
    }

    let resolve = |name: &str| -> FalconResult<Arc<Table>> {
        scope
            .get(name)
            .or_else(|| tables.get(name))
            .cloned()
            .ok_or_else(|| FalconError::InvalidQuery(format!("table '{name}' not found")))
    };

    let table = resolve(&parsed.table)?;

    // Clone the plan so we can move-destructure freely.
    // ParsedQuery is small (no column data), so this is cheap.
    // Names an outer reference can use to point at this query's rows.
    let outer_quals: Vec<String> = std::iter::once(parsed.table.clone())
        .chain(parsed.alias.clone())
        .collect();

    let crate::query::sql::ParsedQuery {
        table: _,
        alias: _,
        select,
        group_by,
        having,
        sort,
        joins,
        set_ops,
        windows,
        ctes: _,
    } = parsed.clone();

    // Cost-based join reordering: put smallest tables first to minimise intermediate sizes.
    let joins = {
        use crate::query::optimizer::reorder_joins;
        reorder_joins(joins, |name| {
            scope.get(name).or_else(|| tables.get(name)).map(|t| t.row_count())
        })
    };

    // A correlated predicate cannot be folded — it is answered per outer row and
    // handled after the scan instead of being pushed into it.
    let mut correlated_filter = None;
    let select = {
        let mut s = select;
        if let Some(f) = s.filter.take() {
            if has_correlated_subquery(&f, tables, &scope)? {
                correlated_filter = Some(f);
            } else {
                // Uncorrelated subqueries have no per-row meaning: run each once.
                s.filter = Some(resolve_subqueries(f, tables, &scope, indices)?);
            }
        }
        s
    };
    let having = having
        .map(|h| resolve_subqueries(h, tables, &scope, indices))
        .transpose()?;
    {

        // Correct SQL execution order:
        //   1. FROM + JOIN
        //   2. WHERE (filter)
        //   3. GROUP BY + HAVING
        //   4. ORDER BY
        //   5. LIMIT / OFFSET
        //   6. SELECT projection (+ DISTINCT)

        let SelectQuery { filter, limit, offset, projection, distinct, scan_columns: _ } = select;

        // Aggregate-only fast path: when there is no WHERE, JOIN, WINDOW, ORDER BY,
        // LIMIT, or DISTINCT, skip execute_select entirely and run the aggregate
        // directly on the raw table batches. Eliminates one full parallel Rayon scan
        // pass and the intermediate Vec<RecordBatch> allocation — roughly 2x lower
        // overhead for simple aggregates like SELECT SUM(score) FROM t.
        if filter.is_none()
            && correlated_filter.is_none()
            && joins.is_empty()
            && windows.is_empty()
            && sort.is_empty()
            && limit.is_none()
            && offset == 0
            && !distinct
            && set_ops.is_empty()
        {
            if let Some(gb) = group_by.as_ref() {
                use crate::query::aggregate::execute_agg_only;
                let result = if gb.group_by.is_empty() {
                    execute_agg_only(&table.batches, &gb.aggs)?
                } else {
                    gb.execute(&table.batches)?
                };
                let mut agg_batches = vec![result];
                if let Some(having_expr) = having {
                    let mut kept = Vec::with_capacity(agg_batches.len());
                    for b in agg_batches {
                        let mask = having_expr.eval_mask_bits(&b)?;
                        let f = b.filter_by_bitmask(&mask)?;
                        if f.row_count() > 0 { kept.push(f); }
                    }
                    agg_batches = kept;
                }
                if let Projection::Columns(cols) = projection {
                    agg_batches = agg_batches.into_iter()
                        .map(|b| project_batch(&b, &cols))
                        .collect::<FalconResult<_>>()?;
                }
                return Ok(agg_batches);
            }
        }

        // Step 1 — scan, applying the WHERE filter only.
        // If we have an index for the filter column, use it to skip unneeded batches.
        let mut scan = SelectQuery::scan_all();

        // Projection pruning for the join's *output*, computed here because `filter`
        // and `correlated_filter` are consumed further down.
        //
        // The scan-level pushdown below is skipped whenever a join is present, since
        // the join needs key columns this analysis does not see. The output is a
        // different question: once the join has matched, only what later stages read
        // has to be built, and building a column costs a cache miss per output row
        // against the larger side. `SELECT COUNT(*)` over a join reads none of them.
        //
        // Applied to the last join only — earlier ones feed the next join's keys,
        // and proving which those are is the analysis that is missing here.
        let join_output_keep: Option<HashSet<String>> = if joins.len() == 1 {
            analyse_needed_columns(
                &projection,
                filter.as_ref(),
                group_by.as_ref(),
                having.as_ref(),
                &sort,
                &windows,
                correlated_filter.as_ref(),
            )
        } else {
            None
        };

        // Projection pushdown: rebuild only the columns some later stage reads.
        // Skipped entirely when a join is involved, because the join needs columns
        // this analysis does not see, and when anything opaque (a subquery, or
        // SELECT *) means the needed set cannot be proven complete.
        if joins.is_empty() {
            if let Some(cols) = needed_columns(
                &projection,
                filter.as_ref(),
                group_by.as_ref(),
                having.as_ref(),
                &sort,
                &windows,
                correlated_filter.as_ref(),
            ) {
                // Only push down when every needed column actually exists here.
                // A reference to a missing column has to reach the normal
                // evaluation path so it reports "column not found"; pruning to an
                // empty batch would turn that error into a silent empty result.
                let all_present = cols
                    .iter()
                    .all(|c| table.schema.fields.iter().any(|f| &f.name == c));
                if all_present {
                    scan = scan.with_scan_columns(cols);
                }
            }
        }

        if let Some(f) = filter { scan = scan.with_filter(f); }
        let table_idx = indices.get(parsed.table.as_str());
        let mut batches = execute_select_with_index(&table, &scan, table_idx)?;

        // Step 1a — correlated predicate: try rewriting to semi/anti join first;
        // fall back to the row-by-row path if the pattern doesn't match.
        if let Some(f) = correlated_filter {
            if let Some((rewrite_joins, residual)) =
                try_rewrite_correlated_to_join(&f, tables, &scope)
            {
                // Execute all semi/anti joins produced by the rewrite.
                for (rj_table, left_key, right_key, anti) in rewrite_joins {
                    let right_table = resolve(&rj_table)?;
                    let right_batches = execute_select(&right_table, &SelectQuery::scan_all())?;
                    let left_k:  Vec<&str> = vec![left_key.as_str()];
                    let right_k: Vec<&str> = vec![right_key.as_str()];
                    let joined = crate::query::join::execute_hash_semi_anti_join(
                        &batches, &right_batches, &left_k, &right_k, anti,
                    )?;
                    batches = if joined.row_count() > 0 { vec![joined] } else { vec![] };
                }
                // Apply any residual filter that could not be rewritten.
                if let Some(rf) = residual {
                    batches = filter_correlated(batches, &rf, tables, &scope, &outer_quals, indices)?;
                }
            } else {
                batches = filter_correlated(batches, &f, tables, &scope, &outer_quals, indices)?;
            }
        }

        // Step 1b — JOINs.
        //
        let last_join = joins.len().saturating_sub(1);
        for (join_idx, join_clause) in joins.iter().enumerate() {
            use crate::query::join::JoinType;
            let right_table = resolve(&join_clause.table)?;
            let right_batches = execute_select(&right_table, &SelectQuery::scan_all())?;
            match join_clause.join_type {
                JoinType::Semi | JoinType::Anti => {
                    let (lk, rk) = extract_join_keys(&join_clause.on)?;
                    let left_keys:  Vec<&str> = lk.iter().map(|s| s.as_str()).collect();
                    let right_keys: Vec<&str> = rk.iter().map(|s| s.as_str()).collect();
                    let anti = join_clause.join_type == JoinType::Anti;
                    let joined = crate::query::join::execute_hash_semi_anti_join(
                        &batches, &right_batches, &left_keys, &right_keys, anti,
                    )?;
                    batches = if joined.row_count() > 0 { vec![joined] } else { vec![] };
                }
                JoinType::Cross => {
                    let joined = crate::query::join::execute_cross_join(
                        &batches, &right_batches,
                    )?;
                    batches = if joined.row_count() > 0 { vec![joined] } else { vec![] };
                }
                _ => {
                    let (lk, rk) = extract_join_keys(&join_clause.on)?;
                    let left_keys:  Vec<&str> = lk.iter().map(|s| s.as_str()).collect();
                    let right_keys: Vec<&str> = rk.iter().map(|s| s.as_str()).collect();
                    let keep = (join_idx == last_join)
                        .then_some(join_output_keep.as_ref())
                        .flatten();
                    let joined = crate::query::join::execute_hash_join_projected(
                        &batches, &right_batches, &left_keys, &right_keys,
                        join_clause.join_type.clone(), keep,
                    )?;
                    batches = if joined.row_count() > 0 { vec![joined] } else { vec![] };
                }
            }
        }

        // Step 2 — GROUP BY + HAVING.
        if let Some(gb) = group_by {
            let result = if gb.group_by.is_empty() {
                execute_agg_only(&batches, &gb.aggs)?
            } else {
                gb.execute(&batches)?
            };
            // HAVING — filter the aggregated result.
            let mut agg_batches = vec![result];
            if let Some(having) = having {
                // Errors here are propagated rather than swallowed. Discarding them
                // silently dropped the batch, so a HAVING clause referring to
                // something the aggregated result does not have — an aggregate that
                // is not in the SELECT list, say — returned zero rows instead of
                // saying what was wrong.
                let mut kept = Vec::with_capacity(agg_batches.len());
                for b in agg_batches {
                    let mask = having.eval_mask_bits(&b)?;
                    let f = b.filter_by_bitmask(&mask)?;
                    if f.row_count() > 0 {
                        kept.push(f);
                    }
                }
                agg_batches = kept;
            }
            batches = agg_batches;
        }

        // Step 2b — Window functions (applied per-batch after GROUP BY).
        if !windows.is_empty() && !batches.is_empty() {
            use crate::query::window::add_window_columns;
            use crate::storage::schema::Field;

            // Merge all batches into one so window functions see the full partition.
            let merged = if batches.len() == 1 {
                batches.remove(0)
            } else {
                crate::query::sort::execute_sort(&batches, &[])?
            };

            let mut extra_fields: Vec<Field>  = Vec::new();
            let mut extra_cols: Vec<crate::storage::column::Column> = Vec::new();

            for win_expr in &windows {
                let col = win_expr.execute(&merged)?;
                let dt = col.data_type();
                extra_fields.push(Field::nullable(&win_expr.alias, dt));
                extra_cols.push(col);
            }

            let with_windows = add_window_columns(merged, extra_fields, extra_cols)?;
            batches = vec![with_windows];
        }

        // Step 3 — ORDER BY, fused with LIMIT where possible.
        //
        // `ORDER BY … LIMIT k` only needs the k best rows, so the sort is told the
        // bound and reorders k rows instead of every row. OFFSET has to be included
        // in the bound — rows that OFFSET skips still have to be ranked first.
        let mut limit_applied = false;
        if !sort.is_empty() && !batches.is_empty() {
            let top_n = limit.map(|k| k.saturating_add(offset));
            let sorted = crate::query::sort::execute_sort_limit(&batches, &sort, top_n)?;
            batches = vec![sorted];
            // The sort kept offset + limit rows; OFFSET still has to be applied.
            limit_applied = top_n.is_some();
        }

        // Step 4 — LIMIT / OFFSET.
        if offset > 0 || (limit.is_some() && !limit_applied) {
            batches = apply_offset_limit(batches, offset, limit);
        }

        // Step 5 — projection (SELECT column list).
        if let Projection::Columns(cols) = projection {
            batches = batches.into_iter()
                .map(|b| project_batch(&b, &cols))
                .collect::<FalconResult<_>>()?;
        }

        // Step 6 — DISTINCT.
        if distinct {
            use crate::query::select::dedup_batches;
            batches = dedup_batches(batches);
        }

        // Step 7 — set operations, applied left to right in written order.
        //
        // Each branch runs through the full pipeline (its own WHERE/GROUP BY/ORDER BY),
        // so `a UNION SELECT dept, COUNT(*) FROM b GROUP BY dept` aggregates the right
        // side before combining, as SQL requires.
        for (op, branch) in set_ops {
            let rhs = execute_plan(&branch, tables, &scope, indices)?;
            batches = apply_set_op(op, batches, rhs);
        }

        Ok(batches)
    }
}

/// The set of columns a query reads, or `None` when that cannot be proven.
///
/// `None` means "scan every column" and is the safe answer. It is returned when
/// the query selects `*`, or when any clause contains a subquery whose column
/// references cannot be enumerated. Returning a partial set instead would prune a
/// column a later stage needs.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn analyse_needed_columns(
    projection: &crate::query::select::Projection,
    filter:     Option<&crate::query::expr::Expr>,
    group_by:   Option<&crate::query::aggregate::GroupByExec>,
    having:     Option<&crate::query::expr::Expr>,
    sort:       &[crate::query::sort::SortExpr],
    windows:    &[crate::query::window::WindowExpr],
    correlated: Option<&crate::query::expr::Expr>,
) -> Option<HashSet<String>> {
    use crate::query::select::Projection;

    let mut cols = HashSet::new();
    let mut ok = true;

    match projection {
        Projection::Columns(exprs) => {
            for e in exprs {
                ok &= e.expr.collect_columns(&mut cols);
            }
        }
        // With GROUP BY (or a bare aggregate) the projection describes the
        // *aggregate's* output, not its input: the result batch is built from the
        // group keys and the aggregate arguments, both walked below. So `SELECT
        // COUNT(*) FROM a JOIN b` reads no column of the join at all — which is the
        // case this exists for.
        _ if group_by.is_some() => {}
        // Otherwise the projection is a wildcard over the input and returns whatever
        // the table has, so nothing can be pruned.
        _ => return None,
    }
    if let Some(f) = filter {
        ok &= f.collect_columns(&mut cols);
    }
    if let Some(h) = having {
        ok &= h.collect_columns(&mut cols);
    }
    // A correlated predicate is evaluated against these rows after the scan.
    if let Some(c) = correlated {
        ok &= c.collect_columns(&mut cols);
    }
    for s in sort {
        ok &= s.expr.collect_columns(&mut cols);
    }
    if let Some(g) = group_by {
        for e in &g.group_by {
            ok &= e.collect_columns(&mut cols);
        }
        for a in &g.aggs {
            ok &= a.input.collect_columns(&mut cols);
        }
    }
    for w in windows {
        if let Some(i) = &w.input {
            ok &= i.collect_columns(&mut cols);
        }
        for e in &w.partition_by {
            ok &= e.collect_columns(&mut cols);
        }
        for s in &w.order_by {
            ok &= s.expr.collect_columns(&mut cols);
        }
    }

    // `None` means "could not be proven", which is not the same as "reads nothing".
    if !ok {
        return None;
    }
    Some(cols)
}

/// Needed columns for a *scan*.
///
/// Same analysis, but an empty set is turned into `None`: pruning a scan to zero
/// columns would leave a batch that reports no rows, and `SELECT 1 FROM t` reads no
/// column while still needing them. Callers that can handle "reads nothing" — the
/// join output, which keeps one column so the row count survives — use
/// [`analyse_needed_columns`] directly.
#[allow(clippy::too_many_arguments)]
fn needed_columns(
    projection: &crate::query::select::Projection,
    filter:     Option<&crate::query::expr::Expr>,
    group_by:   Option<&crate::query::aggregate::GroupByExec>,
    having:     Option<&crate::query::expr::Expr>,
    sort:       &[crate::query::sort::SortExpr],
    windows:    &[crate::query::window::WindowExpr],
    correlated: Option<&crate::query::expr::Expr>,
) -> Option<HashSet<String>> {
    let cols = analyse_needed_columns(
        projection, filter, group_by, having, sort, windows, correlated,
    )?;
    (!cols.is_empty()).then_some(cols)
}

/// An index of every distinct row in `batches`, for membership tests.
///
/// Keyed by row hash; each bucket holds the positions that hashed there so a
/// collision is settled by comparing the rows themselves rather than trusting the
/// hash. This shares `row_hash`/`row_eq` with `dedup_batches`, so DISTINCT, UNION,
/// INTERSECT and EXCEPT all agree on what "the same row" means.
struct RowIndex<'a> {
    batches: &'a [RecordBatch],
    buckets: crate::query::hash::FastMap<u64, Vec<(usize, usize)>>,
}

impl<'a> RowIndex<'a> {
    fn build(batches: &'a [RecordBatch]) -> Self {
        use crate::query::select::row_hash;
        let mut buckets: crate::query::hash::FastMap<u64, Vec<(usize, usize)>> =
            Default::default();
        for (bi, batch) in batches.iter().enumerate() {
            for row in 0..batch.row_count() {
                buckets.entry(row_hash(batch, row)).or_default().push((bi, row));
            }
        }
        Self { batches, buckets }
    }

    fn contains(&self, batch: &RecordBatch, row: usize) -> bool {
        use crate::query::select::{row_eq, row_hash};
        match self.buckets.get(&row_hash(batch, row)) {
            None => false,
            Some(bucket) => bucket
                .iter()
                .any(|&(bi, ri)| row_eq(&self.batches[bi], ri, batch, row)),
        }
    }
}

/// Keep only the rows of `batches` for which `keep_row` holds, deduplicating as it
/// goes. Used by INTERSECT and EXCEPT, which both return a set.
fn retain_distinct(
    batches:  Vec<RecordBatch>,
    keep_row: impl Fn(&RecordBatch, usize) -> bool,
) -> Vec<RecordBatch> {
    use crate::query::select::{row_eq, row_hash};

    // Positions refer to the *input* batches, so a row can be compared against
    // earlier rows of the batch currently being scanned — which has not been
    // filtered or emitted yet.
    let mut seen: crate::query::hash::FastMap<u64, Vec<(usize, usize)>> = Default::default();
    let mut keeps: Vec<Vec<bool>> = Vec::with_capacity(batches.len());

    for (bi, batch) in batches.iter().enumerate() {
        let mut keep = Vec::with_capacity(batch.row_count());
        for row in 0..batch.row_count() {
            if !keep_row(batch, row) {
                keep.push(false);
                continue;
            }
            // Drop duplicates within the left side as well: the result is a set.
            let h = row_hash(batch, row);
            let bucket = seen.entry(h).or_default();
            if bucket.iter().any(|&(pb, pr)| row_eq(&batches[pb], pr, batch, row)) {
                keep.push(false);
            } else {
                bucket.push((bi, row));
                keep.push(true);
            }
        }
        keeps.push(keep);
    }

    let mut out = Vec::with_capacity(batches.len());
    for (batch, keep) in batches.iter().zip(keeps) {
        if keep.iter().any(|&k| k) {
            if let Ok(filtered) = batch.filter_by_mask(&keep) {
                if filtered.row_count() > 0 {
                    out.push(filtered);
                }
            }
        }
    }
    out
}

/// Combine the results of two query blocks under a SQL set operator.
fn apply_set_op(
    op:  crate::query::sql::SetOp,
    lhs: Vec<RecordBatch>,
    rhs: Vec<RecordBatch>,
) -> Vec<RecordBatch> {
    use crate::query::select::dedup_batches;
    use crate::query::sql::SetOp;

    match op {
        SetOp::UnionAll => {
            let mut out = lhs;
            out.extend(rhs);
            out
        }
        SetOp::UnionDistinct => {
            let mut out = lhs;
            out.extend(rhs);
            dedup_batches(out)
        }
        SetOp::Intersect => {
            let right = RowIndex::build(&rhs);
            retain_distinct(lhs, |b, r| right.contains(b, r))
        }
        SetOp::Except => {
            let right = RowIndex::build(&rhs);
            retain_distinct(lhs, |b, r| !right.contains(b, r))
        }
    }
}

/// How many times a recursive CTE's recursive term may run before the query is
/// rejected.
///
/// A recursive term that keeps producing rows — `SELECT n + 1 FROM r` with no
/// stopping predicate — has no fixed point, and every pass materialises another
/// table. The cap turns that into an error the caller can see instead of a hang.
pub const MAX_RECURSIVE_ITERS: usize = 1_000;

/// Rename a set of batches' columns positionally.
///
/// Used to match a recursive CTE's recursive term to the names its anchor term
/// established. Types are left alone; `unify_batch_schemas` widens those.
fn rename_batches_positionally(
    batches: Vec<RecordBatch>,
    names:   &[String],
) -> FalconResult<Vec<RecordBatch>> {
    batches
        .into_iter()
        .map(|batch| {
            if batch.schema.fields.len() != names.len() {
                return Err(FalconError::InvalidQuery(format!(
                    "recursive term returns {} column(s) but the anchor term returns {}",
                    batch.schema.fields.len(), names.len()
                )));
            }
            if batch.schema.fields.iter().zip(names).all(|(f, n)| f.name == *n) {
                return Ok(batch);
            }
            let fields = batch.schema.fields.iter().zip(names)
                .map(|(f, n)| crate::storage::schema::Field::new(n.clone(), f.data_type.clone(), f.nullable))
                .collect();
            RecordBatch::from_shared(
                crate::storage::schema::Schema::new(fields),
                batch.columns.clone(),
            )
        })
        .collect()
}

/// The type both sides of a set operation can be read as.
///
/// Only widening is allowed: an integer can be read as a float, and anything
/// can stand in for a column that was all-NULL. Everything else is a genuine
/// type clash and is reported rather than papered over.
fn unify_data_type(
    a: &crate::storage::types::DataType,
    b: &crate::storage::types::DataType,
) -> Option<crate::storage::types::DataType> {
    use crate::storage::types::DataType as D;
    if a == b {
        return Some(a.clone());
    }
    Some(match (a, b) {
        (D::Null, other) | (other, D::Null) => other.clone(),
        (D::Int32, D::Int64) | (D::Int64, D::Int32) => D::Int64,
        (D::Float32, D::Float64) | (D::Float64, D::Float32) => D::Float64,
        (D::Int32 | D::Int64, D::Float32 | D::Float64)
        | (D::Float32 | D::Float64, D::Int32 | D::Int64) => D::Float64,
        _ => return None,
    })
}

/// Re-read a column under a wider type, per `unify_data_type`.
fn cast_column(
    col: &crate::storage::column::Column,
    to:  &crate::storage::types::DataType,
) -> Option<crate::storage::column::Column> {
    use crate::storage::column::{Column, PrimitiveColumn};
    use crate::storage::types::DataType as D;
    if col.data_type() == *to {
        return Some(col.clone());
    }
    let vals: Vec<crate::storage::types::ScalarValue> =
        (0..col.len()).map(|r| col.get(r).unwrap_or(crate::storage::types::ScalarValue::Null)).collect();
    Some(match to {
        D::Int64   => Column::Int64(PrimitiveColumn::new(vals.iter().map(|v| v.as_i64()).collect())),
        D::Float64 => Column::Float64(PrimitiveColumn::new(vals.iter().map(|v| v.as_f64()).collect())),
        _ => return None,
    })
}

/// Bring a set of result batches under one schema.
///
/// Column *names* come from the first batch positionally — that is the SQL rule
/// for set operations and for recursive CTEs, where the anchor term names the
/// columns and the recursive term is matched by position, not by name. Column
/// *types* are widened to whatever both sides can be read as, because the same
/// expression can yield a different type per branch (`n` is an integer, `n + 1`
/// evaluates as a float).
///
/// Without this, batches from different branches disagree and cannot be stored
/// in a single table.
fn unify_batch_schemas(batches: Vec<RecordBatch>) -> FalconResult<Vec<RecordBatch>> {
    let Some(first) = batches.first() else { return Ok(batches) };
    if batches.iter().all(|b| b.schema == first.schema) {
        return Ok(batches);
    }

    let width = first.schema.fields.len();
    if let Some(bad) = batches.iter().find(|b| b.schema.fields.len() != width) {
        return Err(FalconError::InvalidQuery(format!(
            "set operation branches have different column counts: {width} vs {}",
            bad.schema.fields.len()
        )));
    }

    let mut fields = first.schema.fields.clone();
    for batch in &batches[1..] {
        for (i, field) in fields.iter_mut().enumerate() {
            let other = &batch.schema.fields[i];
            field.data_type = unify_data_type(&field.data_type, &other.data_type)
                .ok_or_else(|| FalconError::InvalidQuery(format!(
                    "set operation branches disagree on the type of column '{}': {:?} vs {:?}",
                    field.name, field.data_type, other.data_type
                )))?;
            field.nullable |= other.nullable;
        }
    }
    let schema = crate::storage::schema::Schema::new(fields);

    batches
        .into_iter()
        .map(|batch| {
            if batch.schema == schema {
                return Ok(batch);
            }
            let columns: Vec<Arc<crate::storage::column::Column>> = batch.columns.iter().zip(&schema.fields)
                .map(|(col, field)| {
                    if col.data_type() == field.data_type {
                        Ok(col.clone())
                    } else {
                        cast_column(col, &field.data_type).map(Arc::new).ok_or_else(|| {
                            FalconError::InvalidQuery(format!(
                                "cannot read column '{}' as {:?}", field.name, field.data_type
                            ))
                        })
                    }
                })
                .collect::<FalconResult<_>>()?;
            RecordBatch::from_shared(schema.clone(), columns)
        })
        .collect()
}

/// Collect result batches into a table so a CTE can be queried like any other.
///
/// The batches are reconciled to one schema first: they can come from different
/// branches of a set operation, and a `Table` holds exactly one schema. An
/// append that still fails after that is a real bug, so it is propagated rather
/// than dropping the rows.
fn table_from_batches(name: &str, batches: Vec<RecordBatch>) -> FalconResult<Table> {
    let batches = unify_batch_schemas(batches)?;
    let schema = batches
        .first()
        .map(|b| b.schema.clone())
        .unwrap_or_else(|| crate::storage::schema::Schema::new(vec![]));

    let mut table = Table::new(name, schema);
    for batch in batches {
        table.append_batch(batch)?;
    }
    Ok(table)
}

/// Keep only the rows where `mask[i] == true`.
fn filter_batch_by_mask(batch: &RecordBatch, mask: &[bool]) -> Option<RecordBatch> {
    let indices: Vec<usize> = mask.iter().enumerate()
        .filter_map(|(i, &keep)| if keep { Some(i) } else { None })
        .collect();
    if indices.is_empty() { return None; }
    let columns: Vec<_> = batch.columns.iter().map(|col| {
        use crate::storage::column::Column;
        use crate::storage::types::ScalarValue;
        let vals: Vec<_> = indices.iter().map(|&r| col.get(r).unwrap_or(ScalarValue::Null)).collect();
        std::sync::Arc::new(match col.as_ref() {
            Column::Int64(_) | Column::Int32(_) =>
                Column::Int64(crate::storage::column::PrimitiveColumn::new(
                    vals.iter().map(|v| v.as_i64()).collect()
                )),
            Column::Float64(_) | Column::Float32(_) =>
                Column::Float64(crate::storage::column::PrimitiveColumn::new(
                    vals.iter().map(|v| v.as_f64()).collect()
                )),
            Column::Utf8(_) =>
                Column::Utf8(crate::storage::column::Utf8Column::new(
                    vals.iter().map(|v| match v { ScalarValue::Utf8(s) => s.clone(), _ => None }).collect()
                )),
            Column::Boolean(_) =>
                Column::Boolean(crate::storage::column::BooleanColumn::new(
                    vals.iter().map(|v| v.as_bool()).collect()
                )),
            _ =>
                Column::Int64(crate::storage::column::PrimitiveColumn::new(
                    vals.iter().map(|v| v.as_i64()).collect()
                )),
        })
    }).collect();
    RecordBatch::from_shared(batch.schema.clone(), columns).ok()
}

/// Replace every `Expr::Subquery` with the scalar it evaluates to.
///
/// A scalar subquery must yield exactly one row and one column; anything else is
/// ambiguous and SQL treats it as an error rather than silently picking a value.
fn resolve_subqueries(
    expr:    crate::query::expr::Expr,
    tables:  &HashMap<String, Arc<Table>>,
    scope:   &HashMap<String, Arc<Table>>,
    indices: &HashMap<String, crate::storage::index::MultiTableRegistry>,
) -> FalconResult<crate::query::expr::Expr> {
    use crate::query::expr::Expr;

    let rec = |e: Expr| resolve_subqueries(e, tables, scope, indices);

    Ok(match expr {
        Expr::Subquery(plan) => {
            let batches = execute_plan(&plan.0, tables, scope, indices)?;
            Expr::Literal(single_scalar(&batches)?)
        }

        Expr::InSubquery { expr, plan, negated } => {
            let batches = execute_plan(&plan.0, tables, scope, indices)?;
            Expr::InList {
                expr: Box::new(rec(*expr)?),
                list: single_column(&batches)?
                    .into_iter()
                    .map(Expr::Literal)
                    .collect(),
                negated,
            }
        }

        Expr::Exists { plan, negated } => {
            let batches = execute_plan(&plan.0, tables, scope, indices)?;
            let has_rows = batches.iter().any(|b| b.row_count() > 0);
            Expr::Literal(crate::storage::types::ScalarValue::Boolean(Some(
                has_rows != negated,
            )))
        }

        Expr::BinaryOp { left, op, right } => Expr::BinaryOp {
            left:  Box::new(rec(*left)?),
            op,
            right: Box::new(rec(*right)?),
        },
        Expr::Not(inner)       => Expr::Not(Box::new(rec(*inner)?)),
        Expr::IsNull(inner)    => Expr::IsNull(Box::new(rec(*inner)?)),
        Expr::IsNotNull(inner) => Expr::IsNotNull(Box::new(rec(*inner)?)),

        Expr::InList { expr, list, negated } => Expr::InList {
            expr:    Box::new(rec(*expr)?),
            list:    list.into_iter().map(rec).collect::<FalconResult<_>>()?,
            negated,
        },

        Expr::Case { when_thens, else_expr } => Expr::Case {
            when_thens: when_thens
                .into_iter()
                .map(|(w, t)| Ok((rec(w)?, rec(t)?)))
                .collect::<FalconResult<_>>()?,
            else_expr: else_expr.map(|e| rec(*e).map(Box::new)).transpose()?,
        },

        other => other,
    })
}

// ─── Correlated subqueries ───────────────────────────────────────────────────
//
// A correlated subquery names a column it cannot resolve itself — it belongs to the
// outer row. Such a query has a different answer per row, so it cannot be folded to
// a constant the way the uncorrelated forms above are.
//
// The strategy is the obvious correct one: evaluate the predicate row by row, binding
// the outer row's values into the subquery before running it. That costs one inner
// execution per outer row. A mature optimiser would instead rewrite the whole thing
// into a join or a grouped aggregate; doing that is future work, and until then a
// correlated subquery over a large outer table will be slow.

/// What a plan can resolve on its own.
#[derive(Default)]
struct PlanScope {
    /// Bare column names from the `FROM` table and anything joined.
    columns: std::collections::HashSet<String>,
    /// Table names and aliases usable as a qualifier (`e` in `e.dept`).
    quals:   std::collections::HashSet<String>,
}

impl PlanScope {
    /// Can this plan resolve `reference` without help from an enclosing query?
    ///
    /// A qualifier matters: with `FROM emp e`, the reference `e.dept` inside a
    /// subquery over `allow` is an *outer* one even though `allow` also has a
    /// `dept` column.
    fn resolves(&self, reference: &str) -> bool {
        match reference.rsplit_once('.') {
            Some((qual, col)) => self.quals.contains(qual) && self.columns.contains(col),
            None => self.columns.contains(reference),
        }
    }
}

fn plan_scope_columns(
    plan:   &crate::query::sql::ParsedQuery,
    tables: &HashMap<String, Arc<Table>>,
    scope:  &HashMap<String, Arc<Table>>,
) -> FalconResult<PlanScope> {
    let mut out = PlanScope::default();

    let mut add = |table_name: &str, alias: Option<&str>| {
        out.quals.insert(table_name.to_string());
        if let Some(a) = alias {
            out.quals.insert(a.to_string());
        }
        if let Some(t) = scope.get(table_name).or_else(|| tables.get(table_name)) {
            for f in &t.schema.fields {
                out.columns.insert(f.name.clone());
            }
        }
    };

    add(&plan.table, plan.alias.as_deref());
    for j in &plan.joins {
        add(&j.table, None);
    }
    for (cte_name, _, _) in &plan.ctes {
        out.quals.insert(cte_name.clone());
    }

    Ok(out)
}

/// Free column references in an expression, not descending into nested subqueries.
fn collect_columns(expr: &crate::query::expr::Expr, out: &mut Vec<String>) {
    use crate::query::expr::Expr;
    match expr {
        Expr::Column(n) => out.push(n.clone()),
        Expr::BinaryOp { left, right, .. } => {
            collect_columns(left, out);
            collect_columns(right, out);
        }
        Expr::Not(e) | Expr::IsNull(e) | Expr::IsNotNull(e) => collect_columns(e, out),
        Expr::Like { expr, .. } => collect_columns(expr, out),
        Expr::InList { expr, list, .. } => {
            collect_columns(expr, out);
            list.iter().for_each(|e| collect_columns(e, out));
        }
        Expr::Case { when_thens, else_expr } => {
            for (w, t) in when_thens {
                collect_columns(w, out);
                collect_columns(t, out);
            }
            if let Some(e) = else_expr {
                collect_columns(e, out);
            }
        }
        Expr::Function { args, .. } => args.iter().for_each(|a| collect_columns(a, out)),
        Expr::Cast { expr, .. } => collect_columns(expr, out),
        _ => {}
    }
}

/// Does this predicate contain a subquery that references the outer row?
fn has_correlated_subquery(
    expr:   &crate::query::expr::Expr,
    tables: &HashMap<String, Arc<Table>>,
    scope:  &HashMap<String, Arc<Table>>,
) -> FalconResult<bool> {
    use crate::query::expr::Expr;

    // Is `plan` unable to resolve some column it mentions?
    let plan_is_correlated = |plan: &crate::query::sql::ParsedQuery| -> FalconResult<bool> {
        let known = plan_scope_columns(plan, tables, scope)?;
        let mut refs = Vec::new();
        if let Some(f) = &plan.select.filter {
            collect_columns(f, &mut refs);
        }
        if let Some(h) = &plan.having {
            collect_columns(h, &mut refs);
        }
        Ok(refs.iter().any(|r| !known.resolves(r)))
    };

    Ok(match expr {
        Expr::Subquery(p) | Expr::Exists { plan: p, .. } => plan_is_correlated(&p.0)?,
        Expr::InSubquery { expr, plan, negated: _ } => {
            plan_is_correlated(&plan.0)? || has_correlated_subquery(expr, tables, scope)?
        }
        Expr::BinaryOp { left, right, .. } => {
            has_correlated_subquery(left, tables, scope)?
                || has_correlated_subquery(right, tables, scope)?
        }
        Expr::Not(e) | Expr::IsNull(e) | Expr::IsNotNull(e) => {
            has_correlated_subquery(e, tables, scope)?
        }
        Expr::InList { expr, list, .. } => {
            if has_correlated_subquery(expr, tables, scope)? {
                true
            } else {
                for e in list {
                    if has_correlated_subquery(e, tables, scope)? {
                        return Ok(true);
                    }
                }
                false
            }
        }
        Expr::Case { when_thens, else_expr } => {
            for (w, t) in when_thens {
                if has_correlated_subquery(w, tables, scope)?
                    || has_correlated_subquery(t, tables, scope)?
                {
                    return Ok(true);
                }
            }
            match else_expr {
                Some(e) => has_correlated_subquery(e, tables, scope)?,
                None => false,
            }
        }
        _ => false,
    })
}

/// Replace `Column(name)` with a literal wherever `bindings` has a value for it.
fn substitute_columns(
    expr:     crate::query::expr::Expr,
    bindings: &HashMap<String, crate::storage::types::ScalarValue>,
) -> crate::query::expr::Expr {
    use crate::query::expr::Expr;
    let rec = |e: Expr| substitute_columns(e, bindings);

    match expr {
        // Match the qualified name first (`e.dept`), then the bare one — a binding
        // for `e.dept` must not be shadowed by an unrelated `dept`.
        Expr::Column(name) => {
            let bare = name.rsplit_once('.').map(|(_, c)| c).unwrap_or(name.as_str());
            match bindings.get(&name).or_else(|| bindings.get(bare)) {
                Some(v) => Expr::Literal(v.clone()),
                None => Expr::Column(name),
            }
        }
        Expr::BinaryOp { left, op, right } => Expr::BinaryOp {
            left:  Box::new(rec(*left)),
            op,
            right: Box::new(rec(*right)),
        },
        Expr::Not(e)       => Expr::Not(Box::new(rec(*e))),
        Expr::IsNull(e)    => Expr::IsNull(Box::new(rec(*e))),
        Expr::IsNotNull(e) => Expr::IsNotNull(Box::new(rec(*e))),
        Expr::InList { expr, list, negated } => Expr::InList {
            expr:    Box::new(rec(*expr)),
            list:    list.into_iter().map(rec).collect(),
            negated,
        },
        Expr::Case { when_thens, else_expr } => Expr::Case {
            when_thens: when_thens.into_iter().map(|(w, t)| (rec(w), rec(t))).collect(),
            else_expr:  else_expr.map(|e| Box::new(rec(*e))),
        },
        other => other,
    }
}

/// Bind the current outer row into every subquery inside `expr`.
///
/// Only names the inner plan cannot resolve itself are replaced — otherwise a column
/// that exists in both scopes (say `id` in both tables) would be wrongly overwritten
/// with the outer value.
fn bind_correlated(
    expr:     crate::query::expr::Expr,
    bindings: &HashMap<String, crate::storage::types::ScalarValue>,
    tables:   &HashMap<String, Arc<Table>>,
    scope:    &HashMap<String, Arc<Table>>,
) -> FalconResult<crate::query::expr::Expr> {
    use crate::query::expr::{Expr, SubqueryPlan};

    let bind_plan = |plan: SubqueryPlan| -> FalconResult<SubqueryPlan> {
        let inner = plan_scope_columns(&plan.0, tables, scope)?;
        // Keep only the names the inner plan cannot resolve itself. A name the
        // subquery can resolve must keep referring to the inner table.
        let outer_only: HashMap<_, _> = bindings
            .iter()
            .filter(|(k, _)| !inner.resolves(k))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let mut p = (*plan.0).clone();
        p.select.filter = p.select.filter.map(|f| substitute_columns(f, &outer_only));
        p.having = p.having.map(|h| substitute_columns(h, &outer_only));
        Ok(SubqueryPlan(Box::new(p)))
    };

    let rec = |e: Expr| bind_correlated(e, bindings, tables, scope);

    Ok(match expr {
        Expr::Subquery(p) => Expr::Subquery(bind_plan(p)?),
        Expr::Exists { plan, negated } => Expr::Exists { plan: bind_plan(plan)?, negated },
        Expr::InSubquery { expr, plan, negated } => Expr::InSubquery {
            expr:    Box::new(rec(*expr)?),
            plan:    bind_plan(plan)?,
            negated,
        },
        Expr::BinaryOp { left, op, right } => Expr::BinaryOp {
            left:  Box::new(rec(*left)?),
            op,
            right: Box::new(rec(*right)?),
        },
        Expr::Not(e)       => Expr::Not(Box::new(rec(*e)?)),
        Expr::IsNull(e)    => Expr::IsNull(Box::new(rec(*e)?)),
        Expr::IsNotNull(e) => Expr::IsNotNull(Box::new(rec(*e)?)),
        Expr::InList { expr, list, negated } => Expr::InList {
            expr:    Box::new(rec(*expr)?),
            list:    list.into_iter().map(rec).collect::<FalconResult<_>>()?,
            negated,
        },
        Expr::Case { when_thens, else_expr } => Expr::Case {
            when_thens: when_thens
                .into_iter()
                .map(|(w, t)| Ok((rec(w)?, rec(t)?)))
                .collect::<FalconResult<_>>()?,
            else_expr: else_expr.map(|e| rec(*e).map(Box::new)).transpose()?,
        },
        other => other,
    })
}

/// Try to rewrite correlated EXISTS/NOT EXISTS/IN filters as semi/anti joins.
///
/// Returns `Some((joins, residual))` where `joins` is a list of
/// `(right_table, left_key, right_key, is_anti)` tuples and `residual` is any
/// part of the filter that could not be rewritten.  Returns `None` if nothing
/// could be rewritten (caller should fall back to row-by-row evaluation).
///
/// Handles two patterns:
///   EXISTS (SELECT 1 FROM t WHERE t.key = outer.key)  → semi-join
///   NOT EXISTS (SELECT 1 FROM t WHERE t.key = outer.key) → anti-join
///   outer.col IN (SELECT col FROM t)  → semi-join on t.col = outer.col
fn try_rewrite_correlated_to_join(
    expr:   &crate::query::expr::Expr,
    tables: &HashMap<String, Arc<Table>>,
    scope:  &HashMap<String, Arc<Table>>,
) -> Option<(Vec<(String, String, String, bool)>, Option<crate::query::expr::Expr>)> {
    use crate::query::expr::Expr;

    // Decompose the top-level AND chain into individual predicates.
    let preds = flatten_and_ref(expr);
    let mut joins:    Vec<(String, String, String, bool)> = Vec::new();
    let mut residual: Vec<crate::query::expr::Expr>      = Vec::new();

    for pred in preds {
        match pred {
            Expr::Exists { plan, negated } => {
                if let Some((tbl, lk, rk)) = extract_equi_join_from_plan(plan, tables, scope) {
                    joins.push((tbl, lk, rk, *negated));
                } else {
                    residual.push(pred.clone());
                }
            }
            Expr::InSubquery { expr, plan, negated } => {
                if let Expr::Column(outer_col) = expr.as_ref() {
                    if let Some(inner_col) = single_column_plan(plan) {
                        joins.push((plan.0.table.clone(), outer_col.clone(), inner_col, *negated));
                    } else {
                        residual.push(pred.clone());
                    }
                } else {
                    residual.push(pred.clone());
                }
            }
            _ => residual.push(pred.clone()),
        }
    }

    if joins.is_empty() {
        return None;
    }

    let residual_expr = if residual.is_empty() {
        None
    } else {
        Some(rebuild_and_exprs(residual))
    };

    Some((joins, residual_expr))
}

/// Extract `(right_table, left_key, right_key)` from a plan whose WHERE clause
/// is of the form `right_table.col = outer_col` (or commuted).
fn extract_equi_join_from_plan(
    plan:   &crate::query::expr::SubqueryPlan,
    tables: &HashMap<String, Arc<Table>>,
    scope:  &HashMap<String, Arc<Table>>,
) -> Option<(String, String, String)> {
    use crate::query::expr::Expr;
    let inner = plan_scope_columns(&plan.0, tables, scope).ok()?;
    let filter = plan.0.select.filter.as_ref()?;

    if let Expr::BinaryOp { left, op: crate::query::expr::BinaryOp::Eq, right } = filter {
        let (lname, rname) = match (left.as_ref(), right.as_ref()) {
            (Expr::Column(a), Expr::Column(b)) => (a.as_str(), b.as_str()),
            _ => return None,
        };
        // One name must be resolvable inside the subquery; the other must not be.
        let (inner_col, outer_col) = if inner.resolves(lname) && !inner.resolves(rname) {
            (bare(lname), bare(rname))
        } else if inner.resolves(rname) && !inner.resolves(lname) {
            (bare(rname), bare(lname))
        } else {
            return None;
        };
        Some((plan.0.table.clone(), outer_col.to_string(), inner_col.to_string()))
    } else {
        None
    }
}

/// If the plan projects exactly one column, return its bare name.
fn single_column_plan(plan: &crate::query::expr::SubqueryPlan) -> Option<String> {
    use crate::query::select::Projection;
    match &plan.0.select.projection {
        Projection::Columns(cols) if cols.len() == 1 => {
            Some(cols[0].alias.clone())
        }
        _ => {
            // SELECT * — use the table's first column name? Not safe without schema.
            None
        }
    }
}

fn bare(name: &str) -> &str {
    name.rsplit_once('.').map(|(_, c)| c).unwrap_or(name)
}

fn flatten_and_ref(expr: &crate::query::expr::Expr) -> Vec<&crate::query::expr::Expr> {
    use crate::query::expr::{BinaryOp, Expr};
    match expr {
        Expr::BinaryOp { left, op: BinaryOp::And, right } => {
            let mut l = flatten_and_ref(left);
            l.extend(flatten_and_ref(right));
            l
        }
        other => vec![other],
    }
}

fn rebuild_and_exprs(mut preds: Vec<crate::query::expr::Expr>) -> crate::query::expr::Expr {
    use crate::query::expr::{BinaryOp, Expr};
    assert!(!preds.is_empty());
    let mut result = preds.remove(0);
    for p in preds {
        result = Expr::BinaryOp {
            left:  Box::new(result),
            op:    BinaryOp::And,
            right: Box::new(p),
        };
    }
    result
}

/// Apply a predicate containing correlated subqueries, one outer row at a time.
fn filter_correlated(
    batches: Vec<RecordBatch>,
    filter:  &crate::query::expr::Expr,
    tables:  &HashMap<String, Arc<Table>>,
    scope:   &HashMap<String, Arc<Table>>,
    quals:   &[String],
    indices: &HashMap<String, crate::storage::index::MultiTableRegistry>,
) -> FalconResult<Vec<RecordBatch>> {
    let mut out = Vec::with_capacity(batches.len());

    for batch in batches {
        let n_rows = batch.row_count();
        let mut keep = Vec::with_capacity(n_rows);

        for row in 0..n_rows {
            // Bind this row's columns, then the subquery is uncorrelated and the
            // normal folding path applies. Each column is bound under its bare name
            // and under every qualifier the outer query can be referenced by, so
            // `e.dept` resolves as readily as `dept`.
            let mut bindings = HashMap::with_capacity(batch.column_count() * (quals.len() + 1));
            for (ci, field) in batch.schema.fields.iter().enumerate() {
                let value = batch.column(ci).get(row)?;
                for q in quals {
                    bindings.insert(format!("{q}.{}", field.name), value.clone());
                }
                bindings.insert(field.name.clone(), value);
            }

            let bound    = bind_correlated(filter.clone(), &bindings, tables, scope)?;
            let resolved = resolve_subqueries(bound, tables, scope, indices)?;

            let row_batch = batch.slice(row, 1)?;
            let mask = resolved.eval_mask_bits(&row_batch)?;
            keep.push(mask.first().map(|w| w & 1 == 1).unwrap_or(false));
        }

        let filtered = batch.filter_by_mask(&keep)?;
        if filtered.row_count() > 0 {
            out.push(filtered);
        }
    }

    Ok(out)
}

/// Collect every value of a single-column result — the right-hand side of `IN (SELECT …)`.
fn single_column(batches: &[RecordBatch]) -> FalconResult<Vec<crate::storage::types::ScalarValue>> {
    let mut out = Vec::new();
    for batch in batches {
        if batch.column_count() != 1 {
            return Err(FalconError::InvalidQuery(format!(
                "IN (SELECT …) requires exactly 1 column, got {}",
                batch.column_count()
            )));
        }
        let col = batch.column(0);
        for row in 0..batch.row_count() {
            out.push(col.get(row)?);
        }
    }
    Ok(out)
}

/// Extract the single value produced by a scalar subquery.
fn single_scalar(batches: &[RecordBatch]) -> FalconResult<crate::storage::types::ScalarValue> {
    use crate::storage::types::ScalarValue;

    let rows: usize = batches.iter().map(|b| b.row_count()).sum();
    if rows == 0 {
        // SQL says an empty scalar subquery is NULL, not an error.
        return Ok(ScalarValue::Null);
    }
    if rows > 1 {
        return Err(FalconError::InvalidQuery(format!(
            "scalar subquery returned {rows} rows, expected 1"
        )));
    }

    let batch = batches.iter().find(|b| b.row_count() > 0).expect("row exists");
    if batch.column_count() != 1 {
        return Err(FalconError::InvalidQuery(format!(
            "scalar subquery returned {} columns, expected 1",
            batch.column_count()
        )));
    }
    batch.column(0).get(0)
}

impl Database {
    // ─── Table management ─────────────────────────────────────────────────────

    pub fn table_names(&self) -> Vec<String> {
        self.tables.read().unwrap().keys().cloned().collect()
    }

    pub fn get_table(&self, name: &str) -> Option<Arc<crate::storage::table::Table>> {
        self.tables.read().unwrap().get(name).cloned()
    }

    pub fn table_info(&self, name: &str) -> Option<TableInfo> {
        let tables = self.tables.read().unwrap();
        tables.get(name).map(|t| TableInfo {
            name:       t.name.clone(),
            row_count:  t.row_count(),
            col_count:  t.schema.field_count(),
            columns:    t.schema.fields.iter()
                .map(|f| ColumnInfo {
                    name:     f.name.clone(),
                    data_type: f.data_type.to_string(),
                    nullable:  f.nullable,
                })
                .collect(),
        })
    }

    pub fn drop_table(&self, name: &str) -> bool {
        self.tables.write().unwrap().remove(name).is_some()
    }
}

// ─── Info structs (for REST / Python) ────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
pub struct TableInfo {
    pub name:      String,
    pub row_count: usize,
    pub col_count: usize,
    pub columns:   Vec<ColumnInfo>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ColumnInfo {
    pub name:      String,
    pub data_type: String,
    pub nullable:  bool,
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Wrap a DML row count as a one-row, one-column result.
fn rows_affected(n: usize) -> QueryResult {
    use crate::storage::{
        column::ColumnBuilder,
        schema::{Field, Schema},
        types::DataType,
    };
    let schema = Schema::new(vec![Field::required("rows_affected", DataType::Int64)]);
    let batch  = RecordBatch::new(schema, vec![ColumnBuilder::int64(vec![Some(n as i64)])])
        .expect("single-column batch is always well-formed");
    QueryResult { batches: vec![batch] }
}

// ─── EXPLAIN ─────────────────────────────────────────────────────────────────

/// Render a duration the way a reader scans it: whole units, three significant
/// figures, no trailing noise.
fn format_duration(d: std::time::Duration) -> String {
    let ns = d.as_nanos();
    if ns < 1_000 {
        format!("{ns}ns")
    } else if ns < 1_000_000 {
        format!("{:.3}µs", ns as f64 / 1_000.0)
    } else if ns < 1_000_000_000 {
        format!("{:.3}ms", ns as f64 / 1_000_000.0)
    } else {
        format!("{:.3}s", ns as f64 / 1_000_000_000.0)
    }
}

fn plan_result(lines: Vec<String>) -> QueryResult {
    use crate::storage::{
        column::{Column, Utf8Column},
        schema::{Field, Schema},
        table::RecordBatch,
        types::DataType,
    };
    let schema = Schema::new(vec![Field::nullable("plan", DataType::Utf8)]);
    let col    = Column::Utf8(Utf8Column::new(lines.into_iter().map(Some).collect()));
    let batch  = RecordBatch::new(schema, vec![col]).expect("single-col batch");
    QueryResult { batches: vec![batch] }
}

impl Database {
    /// `EXPLAIN ANALYZE stmt` — run the statement for real and report the plan
    /// annotated with measured time and the row count it actually produced.
    ///
    /// The point of ANALYZE is the gap between what the planner assumed and what
    /// happened, so this executes the statement rather than estimating. For a
    /// mutating statement that means the mutation is applied: `EXPLAIN ANALYZE
    /// DELETE ...` really deletes. That matches PostgreSQL and DuckDB, where the
    /// usual way to inspect a mutation without keeping it is to wrap it in a
    /// transaction and roll back.
    fn run_explain_analyze(
        &self,
        stmt:     crate::query::sql::ParsedStatement,
        original: &str,
    ) -> FalconResult<QueryResult> {
        use std::time::Instant;

        let plan = explain_statement(&stmt);

        // Re-run through the normal `sql` path so ANALYZE measures the real
        // execution, not a second implementation that could drift from it.
        let inner_sql = strip_explain_prefix(original);
        let started   = Instant::now();
        let outcome   = self.sql(inner_sql);
        let elapsed   = started.elapsed();

        let mut lines: Vec<String> = Vec::new();
        lines.push(format!("Execution time: {}", format_duration(elapsed)));

        match &outcome {
            Ok(r) => {
                lines.push(format!("Rows produced: {}", r.row_count()));
                lines.push(String::new());
            }
            Err(e) => {
                // Report the failure in the plan output rather than discarding the
                // timing: knowing how long a query ran before failing is useful.
                lines.push(format!("Failed after {}: {e}", format_duration(elapsed)));
                lines.push(String::new());
            }
        }

        lines.extend(plan.lines().map(|l| l.to_string()));
        Ok(plan_result(lines))
    }
}

/// Strip a leading `EXPLAIN` / `EXPLAIN ANALYZE` so the inner statement can be
/// handed back to the normal execution path.
fn strip_explain_prefix(sql: &str) -> &str {
    let trimmed = sql.trim_start();
    let rest = match trimmed.get(..7) {
        Some(p) if p.eq_ignore_ascii_case("explain") => trimmed[7..].trim_start(),
        _ => return trimmed,
    };
    match rest.get(..7) {
        Some(p) if p.eq_ignore_ascii_case("analyze") => rest[7..].trim_start(),
        _ => rest,
    }
}

fn run_explain(stmt: crate::query::sql::ParsedStatement) -> QueryResult {
    use crate::storage::{
        column::{Column, Utf8Column},
        schema::{Field, Schema},
        table::RecordBatch,
        types::DataType,
    };

    let plan_text = explain_statement(&stmt);
    let lines: Vec<Option<String>> = plan_text
        .lines()
        .map(|l| Some(l.to_string()))
        .collect();

    let schema = Schema::new(vec![Field::nullable("plan", DataType::Utf8)]);
    let col    = Column::Utf8(Utf8Column::new(lines));
    let batch  = RecordBatch::new(schema, vec![col]).expect("single-col batch");
    QueryResult { batches: vec![batch] }
}

fn explain_statement(stmt: &crate::query::sql::ParsedStatement) -> String {
    use crate::query::sql::ParsedStatement;
    match stmt {
        ParsedStatement::Query(q)   => explain_query(q),
        ParsedStatement::Insert(ins) => format!(
            "Insert into {} ({})",
            ins.table,
            match &ins.source {
                crate::query::dml::InsertSource::Values(rows) =>
                    format!("{} rows", rows.len()),
                crate::query::dml::InsertSource::Select(_) =>
                    "SELECT".to_string(),
            }
        ),
        ParsedStatement::Update(upd) => format!(
            "Update {} SET {} columns{}",
            upd.table,
            upd.assignments.len(),
            if upd.filter.is_some() { " (filtered)" } else { "" }
        ),
        ParsedStatement::Delete(del) => format!(
            "Delete from {}{}",
            del.table,
            if del.filter.is_some() { " (filtered)" } else { "" }
        ),
        ParsedStatement::CreateTable(ct) => {
            let cols = ct.columns.iter()
                .map(|c| {
                    let nn = if !c.nullable { " NOT NULL" } else { "" };
                    format!("{} {}{}", c.name, c.data_type, nn)
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("CreateTable: {} ({})", ct.name, cols)
        }
        ParsedStatement::DropTable { name, if_exists } => {
            if *if_exists {
                format!("DropTable: {name} (IF EXISTS)")
            } else {
                format!("DropTable: {name}")
            }
        }
        ParsedStatement::AlterTable(alt) => {
            use crate::query::sql::AlterAction;
            let action = match &alt.action {
                AlterAction::AddColumn(c) => format!("ADD COLUMN {} {}", c.name, c.data_type),
                AlterAction::DropColumn(n) => format!("DROP COLUMN {n}"),
                AlterAction::AddConstraint(tc) => format!("ADD CONSTRAINT {:?}", tc.name),
                AlterAction::DropConstraint(n) => format!("DROP CONSTRAINT {n}"),
            };
            format!("AlterTable: {} — {action}", alt.table)
        }
        ParsedStatement::Explain { stmt, analyze } => {
            let kw = if *analyze { "Explain Analyze" } else { "Explain" };
            format!("{kw}:\n{}", explain_statement(stmt))
        }
        ParsedStatement::Copy { table, path, format } => {
            format!("Copy: {table} <- {path} ({format:?})")
        }
        ParsedStatement::Vacuum { table } => match table {
            Some(t) => format!("Vacuum: {t}"),
            None    => "Vacuum: all tables".to_string(),
        },
        ParsedStatement::Begin    => "BEGIN".to_string(),
        ParsedStatement::Commit   => "COMMIT".to_string(),
        ParsedStatement::Rollback => "ROLLBACK".to_string(),
        ParsedStatement::Truncate { name } => format!("Truncate: {name}"),
        ParsedStatement::Checkpoint { path } => match path {
            Some(p) => format!("Checkpoint to '{p}'"),
            None    => "Checkpoint (WAL dir)".to_string(),
        },
        ParsedStatement::CreateIndex(ci) => format!(
            "CreateIndex: {} ON {} ({}) USING {:?}{}",
            ci.name, ci.table, ci.column,
            ci.index_type,
            if ci.unique { " UNIQUE" } else { "" }
        ),
        ParsedStatement::DropIndex { name, if_exists } => {
            if *if_exists { format!("DropIndex: {name} (IF EXISTS)") }
            else          { format!("DropIndex: {name}") }
        }
    }
}

fn explain_query(q: &crate::query::sql::ParsedQuery) -> String {
    use crate::query::{
        aggregate::AggFunc,
        join::JoinType,
        sort::SortOrder,
    };

    let mut lines: Vec<String> = Vec::new();

    // CTEs
    for (name, _, _) in &q.ctes {
        lines.push(format!("CTE: {name}"));
    }

    // FROM + JOINs
    if q.joins.is_empty() {
        lines.push(format!("Scan: {}", q.table));
    } else {
        let join_parts: Vec<_> = q.joins.iter().map(|j| {
            let kind = match j.join_type {
                JoinType::Inner => "INNER JOIN",
                JoinType::Left  => "LEFT JOIN",
                JoinType::Right => "RIGHT JOIN",
                JoinType::Full  => "FULL OUTER JOIN",
                JoinType::Cross => "CROSS JOIN",
                JoinType::Semi  => "SEMI JOIN",
                JoinType::Anti  => "ANTI JOIN",
            };
            format!("{kind} {}", j.table)
        }).collect();
        lines.push(format!("Scan: {} + {}", q.table, join_parts.join(", ")));
    }

    // WHERE
    if let Some(f) = &q.select.filter {
        lines.push(format!("Filter: {f}"));
    }

    // GROUP BY + aggregates
    if let Some(gb) = &q.group_by {
        let key_count = gb.group_by.len();
        let agg_parts: Vec<_> = gb.aggs.iter().map(|a| {
            let func = match &a.func {
                AggFunc::Count         => "COUNT".to_string(),
                AggFunc::CountDistinct => "COUNT(DISTINCT)".to_string(),
                AggFunc::Sum           => "SUM".to_string(),
                AggFunc::Avg           => "AVG".to_string(),
                AggFunc::Min           => "MIN".to_string(),
                AggFunc::Max           => "MAX".to_string(),
                AggFunc::StringAgg { separator } => format!("STRING_AGG('{separator}')"),
                AggFunc::BoolAnd       => "BOOL_AND".to_string(),
                AggFunc::BoolOr        => "BOOL_OR".to_string(),
                AggFunc::Stddev { population: true  } => "STDDEV_POP".to_string(),
                AggFunc::Stddev { population: false } => "STDDEV".to_string(),
                AggFunc::Variance { population: true  } => "VAR_POP".to_string(),
                AggFunc::Variance { population: false } => "VARIANCE".to_string(),
                AggFunc::First         => "FIRST".to_string(),
                AggFunc::Last          => "LAST".to_string(),
            };
            format!("{func}({}) as {}", a.input, a.alias)

        }).collect();
        lines.push(format!(
            "GroupBy: {} key(s) | {}",
            key_count,
            agg_parts.join(", ")
        ));
    }

    // Window functions
    if !q.windows.is_empty() {
        let win_parts: Vec<_> = q.windows.iter()
            .map(|w| format!("{:?} as {}", w.func, w.alias))
            .collect();
        lines.push(format!("Window: {}", win_parts.join(", ")));
    }

    // ORDER BY
    if !q.sort.is_empty() {
        let sort_parts: Vec<_> = q.sort.iter().map(|s| {
            let dir = match s.order { SortOrder::Asc => "ASC", SortOrder::Desc => "DESC" };
            format!("{} {dir}", s.expr)
        }).collect();
        lines.push(format!("Sort: {}", sort_parts.join(", ")));
    }

    // LIMIT / OFFSET
    if let Some(lim) = q.select.limit {
        lines.push(format!("Limit: {lim}"));
    }
    if q.select.offset > 0 {
        lines.push(format!("Offset: {}", q.select.offset));
    }

    // DISTINCT
    if q.select.distinct {
        lines.push("Distinct".to_string());
    }

    // Set operations, in the order they are applied.
    for (op, branch) in &q.set_ops {
        lines.push(format!("{op}"));
        for line in explain_query(branch).lines() {
            lines.push(format!("  {line}"));
        }
    }

    lines.join("\n")
}

// ─── Helpers (DDL) ────────────────────────────────────────────────────────────

fn make_null_column(dt: &crate::storage::types::DataType, n: usize) -> crate::storage::column::Column {
    use crate::storage::{
        column::{BooleanColumn, Column, Decimal128Column, PrimitiveColumn, Utf8Column},
        types::DataType,
    };
    match dt {
        DataType::Boolean  => Column::Boolean(BooleanColumn::new(vec![None; n])),
        DataType::Int32    => Column::Int32(PrimitiveColumn::<i32>::new(vec![None; n])),
        DataType::Int64    => Column::Int64(PrimitiveColumn::<i64>::new(vec![None; n])),
        DataType::Float32  => Column::Float32(PrimitiveColumn::<f32>::new(vec![None; n])),
        DataType::Float64  => Column::Float64(PrimitiveColumn::<f64>::new(vec![None; n])),
        DataType::Date32   => Column::Date32(PrimitiveColumn::<i32>::new(vec![None; n])),
        DataType::Timestamp => Column::Timestamp(PrimitiveColumn::<i64>::new(vec![None; n])),
        DataType::Decimal128(p, s) => Column::Decimal128(
            Decimal128Column::new(vec![None; n], *p, *s)
        ),
        DataType::Utf8 | DataType::Null => {
            Column::Utf8(Utf8Column::new(vec![None::<String>; n]))
        }
    }
}

/// Parse a JOIN ON equi-condition into (left_keys, right_keys).
/// Run a SELECT, using the index for equality/range predicates when available.
fn execute_select_with_index(
    table:   &Arc<crate::storage::table::Table>,
    scan:    &crate::query::select::SelectQuery,
    reg:     Option<&crate::storage::index::MultiTableRegistry>,
) -> FalconResult<Vec<RecordBatch>> {
    use crate::query::expr::{BinaryOp, Expr};

    // Try to extract a simple `col = literal` or `col BETWEEN lo AND hi` predicate.
    let maybe_index_result: Option<Vec<RecordBatch>> = (|| {
        let filter = scan.filter.as_ref()?;
        let reg = reg?;

        match filter {
            // col = literal or literal = col — try hash/btree equality
            Expr::BinaryOp { left, op: BinaryOp::Eq, right } => {
                let (col, lit) = if let (Expr::Column(c), Expr::Literal(v)) =
                    (left.as_ref(), right.as_ref()) { (c, v) }
                else if let (Expr::Literal(v), Expr::Column(c)) =
                    (left.as_ref(), right.as_ref()) { (c, v) }
                else { return None };
                let idx = reg.get(col)?;
                let hits = idx.lookup_eq(lit);
                if hits.is_empty() { return Some(Vec::new()); }
                collect_index_hits(table, &hits).ok()
            }
            // col BETWEEN lo AND hi — try btree range
            Expr::BinaryOp {
                left: lo_expr,
                op: BinaryOp::And,
                right: hi_expr,
            } => {
                // Pattern: (col >= lo) AND (col <= hi)
                let (col, lo, hi) = match (lo_expr.as_ref(), hi_expr.as_ref()) {
                    (
                        Expr::BinaryOp { left: c1, op: BinaryOp::GtEq, right: lo },
                        Expr::BinaryOp { left: c2, op: BinaryOp::LtEq, right: hi },
                    ) => {
                        let col1 = match c1.as_ref() { Expr::Column(c) => c, _ => return None };
                        let col2 = match c2.as_ref() { Expr::Column(c) => c, _ => return None };
                        if col1 != col2 { return None; }
                        let lo = match lo.as_ref() { Expr::Literal(v) => v, _ => return None };
                        let hi = match hi.as_ref() { Expr::Literal(v) => v, _ => return None };
                        (col1.as_str(), lo, hi)
                    }
                    _ => return None,
                };
                let idx = reg.get(col)?;
                let hits = idx.lookup_range(lo, hi);
                if hits.is_empty() { return Some(Vec::new()); }
                collect_index_hits(table, &hits).ok()
            }
            _ => None,
        }
    })();

    match maybe_index_result {
        Some(batches) => Ok(batches),
        None => execute_select(table, scan),
    }
}

/// Gather rows identified by (batch_idx, row_idx) pairs from the index.
fn collect_index_hits(
    table: &crate::storage::table::Table,
    hits:  &[(usize, usize)],
) -> FalconResult<Vec<RecordBatch>> {
    use std::collections::HashMap as HM;

    // Group rows by batch to minimise the number of filter operations.
    let mut by_batch: HM<usize, Vec<usize>> = HM::new();
    for &(batch_idx, row_idx) in hits {
        by_batch.entry(batch_idx).or_default().push(row_idx);
    }

    let mut out = Vec::new();
    for (batch_idx, mut rows) in by_batch {
        if batch_idx >= table.batches.len() { continue; }
        let batch = &table.batches[batch_idx];
        rows.sort_unstable();
        rows.dedup();
        let filtered = batch.filter_by_indices(&rows)?;
        if filtered.row_count() > 0 {
            out.push(filtered);
        }
    }
    Ok(out)
}

/// Handles `a.col = b.col`, `col1 = col2`, and AND-chains of those.
fn extract_join_keys(
    on: &crate::query::expr::Expr,
) -> FalconResult<(Vec<String>, Vec<String>)> {
    use crate::query::expr::{BinaryOp, Expr};

    match on {
        Expr::BinaryOp { left, op: BinaryOp::Eq, right } => {
            let l = col_name_from_expr(left)?;
            let r = col_name_from_expr(right)?;
            Ok((vec![l], vec![r]))
        }
        Expr::BinaryOp { left, op: BinaryOp::And, right } => {
            let (mut lk, mut rk) = extract_join_keys(left)?;
            let (lk2, rk2) = extract_join_keys(right)?;
            lk.extend(lk2);
            rk.extend(rk2);
            Ok((lk, rk))
        }
        other => Err(FalconError::InvalidQuery(
            format!("JOIN ON clause must be equi-condition, got: {other:?}")
        )),
    }
}

fn col_name_from_expr(e: &crate::query::expr::Expr) -> FalconResult<String> {
    use crate::query::expr::Expr;
    match e {
        Expr::Column(name) => {
            // Strip table qualifier: "t1.col" → "col"
            Ok(name.split('.').last().unwrap_or(name).to_string())
        }
        other => Err(FalconError::InvalidQuery(
            format!("JOIN key must be a column reference, got: {other:?}")
        )),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn csv_db(content: &str) -> (Database, NamedTempFile) {
        let mut f = NamedTempFile::new().unwrap();
        write!(f, "{content}").unwrap();
        let db = Database::in_memory();
        db.load_csv(f.path(), "t").unwrap();
        (db, f)
    }

    #[test]
    fn sql_select_star() {
        let (db, _f) = csv_db("id,name\n1,alice\n2,bob\n3,carol\n");
        let result = db.sql("SELECT * FROM t").unwrap();
        assert_eq!(result.row_count(), 3);
    }

    #[test]
    fn sql_where_filter() {
        let (db, _f) = csv_db("id,score\n1,9.5\n2,7.0\n3,8.5\n");
        let result = db.sql("SELECT * FROM t WHERE score > 8.0").unwrap();
        assert_eq!(result.row_count(), 2);
    }

    #[test]
    fn sql_limit() {
        let (db, _f) = csv_db("id,name\n1,a\n2,b\n3,c\n4,d\n5,e\n");
        let result = db.sql("SELECT * FROM t LIMIT 3").unwrap();
        assert_eq!(result.row_count(), 3);
    }

    #[test]
    fn sql_order_by() {
        let (db, _f) = csv_db("name,score\nalice,9.5\nbob,7.0\ncarol,8.5\n");
        let result = db.sql("SELECT * FROM t ORDER BY score DESC").unwrap();
        let rows   = result.to_rows();
        assert_eq!(rows[0][0], "alice");
    }

    #[test]
    fn sql_group_by_count() {
        let (db, _f) = csv_db("dept,salary\neng,100\nhr,80\neng,120\nhr,90\n");
        let result = db.sql("SELECT dept, COUNT(*) as cnt FROM t GROUP BY dept").unwrap();
        assert_eq!(result.row_count(), 2);
    }

    #[test]
    fn sql_group_by_sum() {
        let (db, _f) = csv_db("dept,salary\neng,100\nhr,80\neng,120\n");
        let result = db.sql("SELECT dept, SUM(salary) as total FROM t GROUP BY dept").unwrap();
        assert_eq!(result.row_count(), 2);
        let rows = result.to_rows();
        let eng_row = rows.iter().find(|r| r[0] == "eng").unwrap();
        assert_eq!(eng_row[1], "220");
    }

    #[test]
    fn table_not_found() {
        let db = Database::in_memory();
        assert!(db.sql("SELECT * FROM missing").is_err());
    }

    #[test]
    fn table_info() {
        let (db, _f) = csv_db("id,name,score\n1,alice,9.5\n");
        let info = db.table_info("t").unwrap();
        assert_eq!(info.row_count, 1);
        assert_eq!(info.col_count, 3);
    }

    #[test]
    fn sql_window_row_number() {
        let (db, _f) = csv_db("dept,salary\neng,100\neng,120\nhr,80\nhr,90\n");
        let result = db.sql(
            "SELECT dept, salary, ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) as rn FROM t"
        ).unwrap();
        assert_eq!(result.row_count(), 4);
        let cols = result.column_names();
        assert!(cols.contains(&"rn".to_string()), "expected 'rn' column, got: {:?}", cols);
        // eng salaries 100, 120 → sorted desc → 120(rn=1), 100(rn=2)
        // hr salaries 80, 90 → sorted desc → 90(rn=1), 80(rn=2)
        let rows = result.to_rows();
        for row in &rows {
            let dept   = &row[0];
            let salary = row[1].parse::<i64>().unwrap();
            let rn     = row[2].parse::<i64>().unwrap();
            match dept.as_str() {
                "eng" => {
                    if salary == 120 { assert_eq!(rn, 1); }
                    if salary == 100 { assert_eq!(rn, 2); }
                }
                "hr" => {
                    if salary == 90 { assert_eq!(rn, 1); }
                    if salary == 80 { assert_eq!(rn, 2); }
                }
                _ => panic!("unexpected dept {dept}"),
            }
        }
    }

    #[test]
    fn sql_window_rank_ties() {
        let (db, _f) = csv_db("score\n10\n10\n8\n");
        // SELECT score, RANK() OVER (ORDER BY score DESC) as r
        // columns: [score, r]
        let result = db.sql(
            "SELECT score, RANK() OVER (ORDER BY score DESC) as r FROM t"
        ).unwrap();
        assert_eq!(result.row_count(), 3);
        let cols = result.column_names();
        assert_eq!(cols, vec!["score", "r"], "unexpected columns: {:?}", cols);
        let rows = result.to_rows();
        let tied_ranks: Vec<i64> = rows.iter()
            .filter(|r| r[0] == "10")
            .map(|r| r[1].parse().unwrap())
            .collect();
        assert!(tied_ranks.iter().all(|&r| r == 1), "tied 10s should both be rank 1");
        let rank_8: i64 = rows.iter()
            .find(|r| r[0] == "8")
            .unwrap()[1].parse().unwrap();
        assert_eq!(rank_8, 3, "8 should be rank 3 (gap after two ties)");
    }

    #[test]
    fn save_and_load_fdb() {
        let dir  = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("t.fdb");

        let (db, _f) = csv_db("id,score\n1,9.5\n2,8.0\n3,7.5\n");
        db.save_fdb("t", &path).unwrap();

        let db2 = Database::in_memory();
        db2.load_fdb(&path, "t2").unwrap();
        let result = db2.sql("SELECT * FROM t2").unwrap();
        assert_eq!(result.row_count(), 3);
    }

    // ─── Constraint tests ─────────────────────────────────────────────────────

    #[test]
    fn not_null_rejects_null_insert() {
        let db = Database::in_memory();
        db.sql("CREATE TABLE t (id INT NOT NULL, name TEXT)").unwrap();
        let err = db.sql("INSERT INTO t VALUES (NULL, 'alice')").err().unwrap();
        assert!(err.to_string().contains("NOT NULL"), "{err}");
    }

    #[test]
    fn not_null_allows_non_null_insert() {
        let db = Database::in_memory();
        db.sql("CREATE TABLE t (id INT NOT NULL)").unwrap();
        db.sql("INSERT INTO t VALUES (1)").unwrap();
        db.sql("INSERT INTO t VALUES (2)").unwrap();
        assert_eq!(db.sql("SELECT id FROM t").unwrap().row_count(), 2);
    }

    #[test]
    fn not_null_rejects_null_update() {
        let db = Database::in_memory();
        db.sql("CREATE TABLE t (id INT NOT NULL)").unwrap();
        db.sql("INSERT INTO t VALUES (1)").unwrap();
        let err = db.sql("UPDATE t SET id = NULL WHERE id = 1").err().unwrap();
        assert!(err.to_string().contains("NOT NULL"), "{err}");
    }

    #[test]
    fn primary_key_rejects_duplicate() {
        let db = Database::in_memory();
        db.sql("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)").unwrap();
        db.sql("INSERT INTO t VALUES (1, 'alice')").unwrap();
        let err = db.sql("INSERT INTO t VALUES (1, 'bob')").err().unwrap();
        assert!(err.to_string().contains("PRIMARY KEY"), "{err}");
    }

    #[test]
    fn primary_key_rejects_null() {
        let db = Database::in_memory();
        db.sql("CREATE TABLE t (id INT PRIMARY KEY)").unwrap();
        let err = db.sql("INSERT INTO t VALUES (NULL)").err().unwrap();
        assert!(err.to_string().contains("NOT NULL") || err.to_string().contains("PRIMARY KEY"), "{err}");
    }

    #[test]
    fn unique_rejects_duplicate() {
        let db = Database::in_memory();
        db.sql("CREATE TABLE t (email TEXT UNIQUE, x INT)").unwrap();
        db.sql("INSERT INTO t VALUES ('a@b.com', 1)").unwrap();
        let err = db.sql("INSERT INTO t VALUES ('a@b.com', 2)").err().unwrap();
        assert!(err.to_string().contains("UNIQUE"), "{err}");
    }

    #[test]
    fn unique_allows_multiple_nulls() {
        let db = Database::in_memory();
        db.sql("CREATE TABLE t (email TEXT UNIQUE, x INT)").unwrap();
        db.sql("INSERT INTO t VALUES (NULL, 1)").unwrap();
        db.sql("INSERT INTO t VALUES (NULL, 2)").unwrap();
        assert_eq!(db.sql("SELECT x FROM t").unwrap().row_count(), 2);
    }

    #[test]
    fn check_constraint_rejects_violation() {
        let db = Database::in_memory();
        db.sql("CREATE TABLE t (age INT CHECK (age > 0))").unwrap();
        let err = db.sql("INSERT INTO t VALUES (-1)").err().unwrap();
        assert!(err.to_string().contains("CHECK"), "{err}");
    }

    #[test]
    fn check_constraint_allows_valid_row() {
        let db = Database::in_memory();
        db.sql("CREATE TABLE t (age INT CHECK (age > 0))").unwrap();
        db.sql("INSERT INTO t VALUES (25)").unwrap();
        assert_eq!(db.sql("SELECT age FROM t").unwrap().row_count(), 1);
    }

    #[test]
    fn check_null_passes_constraint() {
        let db = Database::in_memory();
        db.sql("CREATE TABLE t (age INT CHECK (age > 0))").unwrap();
        // NULL does not violate CHECK (SQL three-valued logic)
        db.sql("INSERT INTO t VALUES (NULL)").unwrap();
        assert_eq!(db.sql("SELECT age FROM t").unwrap().row_count(), 1);
    }

    #[test]
    fn table_level_primary_key() {
        let db = Database::in_memory();
        db.sql("CREATE TABLE t (id INT, name TEXT, PRIMARY KEY (id))").unwrap();
        db.sql("INSERT INTO t VALUES (1, 'alice')").unwrap();
        let err = db.sql("INSERT INTO t VALUES (1, 'bob')").err().unwrap();
        assert!(err.to_string().contains("PRIMARY KEY"), "{err}");
    }

    #[test]
    fn update_violating_unique_is_rejected() {
        let db = Database::in_memory();
        db.sql("CREATE TABLE t (id INT PRIMARY KEY, v INT)").unwrap();
        db.sql("INSERT INTO t VALUES (1, 10)").unwrap();
        db.sql("INSERT INTO t VALUES (2, 20)").unwrap();
        let err = db.sql("UPDATE t SET id = 1 WHERE id = 2").err().unwrap();
        assert!(err.to_string().contains("PRIMARY KEY"), "{err}");
    }

    // ─── Transaction tests ────────────────────────────────────────────────────

    #[test]
    fn txn_commit_makes_inserts_visible() {
        let db = Database::in_memory();
        db.sql("CREATE TABLE t (x INT)").unwrap();
        db.sql("BEGIN").unwrap();
        db.sql("INSERT INTO t VALUES (1)").unwrap();
        db.sql("INSERT INTO t VALUES (2)").unwrap();
        db.sql("COMMIT").unwrap();
        let r = db.sql("SELECT x FROM t").unwrap();
        assert_eq!(r.row_count(), 2);
    }

    #[test]
    fn txn_rollback_restores_inserts() {
        let db = Database::in_memory();
        db.sql("CREATE TABLE t (x INT)").unwrap();
        db.sql("INSERT INTO t VALUES (1)").unwrap();
        db.sql("BEGIN").unwrap();
        db.sql("INSERT INTO t VALUES (2)").unwrap();
        db.sql("INSERT INTO t VALUES (3)").unwrap();
        db.sql("ROLLBACK").unwrap();
        let r = db.sql("SELECT x FROM t").unwrap();
        assert_eq!(r.row_count(), 1);
    }

    #[test]
    fn txn_rollback_restores_update() {
        let db = Database::in_memory();
        db.sql("CREATE TABLE t (x INT)").unwrap();
        db.sql("INSERT INTO t VALUES (10)").unwrap();
        db.sql("BEGIN").unwrap();
        db.sql("UPDATE t SET x = 99 WHERE x = 10").unwrap();
        let mid = db.sql("SELECT x FROM t").unwrap();
        assert_eq!(mid.to_rows()[0][0], "99");
        db.sql("ROLLBACK").unwrap();
        let r = db.sql("SELECT x FROM t").unwrap();
        assert_eq!(r.to_rows()[0][0], "10");
    }

    #[test]
    fn txn_rollback_restores_delete() {
        let db = Database::in_memory();
        db.sql("CREATE TABLE t (x INT)").unwrap();
        db.sql("INSERT INTO t VALUES (1)").unwrap();
        db.sql("INSERT INTO t VALUES (2)").unwrap();
        db.sql("BEGIN").unwrap();
        db.sql("DELETE FROM t WHERE x = 1").unwrap();
        db.sql("ROLLBACK").unwrap();
        let r = db.sql("SELECT x FROM t").unwrap();
        assert_eq!(r.row_count(), 2);
    }

    #[test]
    fn txn_rollback_undoes_create_table() {
        let db = Database::in_memory();
        db.sql("BEGIN").unwrap();
        db.sql("CREATE TABLE temp_tbl (x INT)").unwrap();
        db.sql("ROLLBACK").unwrap();
        let err = db.sql("SELECT x FROM temp_tbl").err().unwrap();
        assert!(err.to_string().contains("not found"), "{err}");
    }

    #[test]
    fn txn_nested_begin_errors() {
        let db = Database::in_memory();
        db.sql("BEGIN").unwrap();
        let err = db.sql("BEGIN").err().unwrap();
        assert!(err.to_string().contains("transaction"), "{err}");
        db.sql("ROLLBACK").unwrap();
    }

    #[test]
    fn txn_commit_without_begin_errors() {
        let db = Database::in_memory();
        let err = db.sql("COMMIT").err().unwrap();
        assert!(err.to_string().contains("transaction"), "{err}");
    }

    #[test]
    fn txn_rollback_without_begin_errors() {
        let db = Database::in_memory();
        let err = db.sql("ROLLBACK").err().unwrap();
        assert!(err.to_string().contains("transaction"), "{err}");
    }

    // ─── CREATE / DROP INDEX ──────────────────────────────────────────────────

    #[test]
    fn create_btree_index_and_lookup() {
        let db = Database::in_memory();
        db.sql("CREATE TABLE emp (id INT NOT NULL, name TEXT)").unwrap();
        db.sql("INSERT INTO emp VALUES (1, 'alice')").unwrap();
        db.sql("INSERT INTO emp VALUES (2, 'bob')").unwrap();
        db.sql("INSERT INTO emp VALUES (3, 'carol')").unwrap();
        db.sql("CREATE INDEX idx_id ON emp (id)").unwrap();
        let r = db.sql("SELECT name FROM emp WHERE id = 2").unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(r.to_rows()[0][0], "bob");
    }

    #[test]
    fn btree_index_range_lookup() {
        let db = Database::in_memory();
        db.sql("CREATE TABLE t (x INT)").unwrap();
        for i in 1..=10 {
            db.sql(&format!("INSERT INTO t VALUES ({i})")).unwrap();
        }
        db.sql("CREATE INDEX idx_x ON t (x)").unwrap();
        // BETWEEN uses range index lookup.
        let r = db.sql("SELECT x FROM t WHERE x BETWEEN 3 AND 5").unwrap();
        assert_eq!(r.row_count(), 3);
    }

    #[test]
    fn drop_index_removes_it() {
        let db = Database::in_memory();
        db.sql("CREATE TABLE t (x INT)").unwrap();
        db.sql("INSERT INTO t VALUES (1)").unwrap();
        db.sql("CREATE INDEX idx ON t (x)").unwrap();
        db.sql("DROP INDEX idx").unwrap();
        // After drop, query should still work (falls back to full scan).
        let r = db.sql("SELECT x FROM t WHERE x = 1").unwrap();
        assert_eq!(r.row_count(), 1);
    }

    #[test]
    fn index_if_not_exists() {
        let db = Database::in_memory();
        db.sql("CREATE TABLE t (x INT)").unwrap();
        db.sql("CREATE INDEX idx ON t (x)").unwrap();
        db.sql("CREATE INDEX IF NOT EXISTS idx ON t (x)").unwrap(); // must not error
    }

    #[test]
    fn index_rebuilt_after_insert() {
        let db = Database::in_memory();
        db.sql("CREATE TABLE t (x INT)").unwrap();
        db.sql("CREATE INDEX idx ON t (x)").unwrap();
        db.sql("INSERT INTO t VALUES (42)").unwrap();
        db.sql("INSERT INTO t VALUES (99)").unwrap();
        let r = db.sql("SELECT x FROM t WHERE x = 42").unwrap();
        assert_eq!(r.row_count(), 1);
    }

    // ─── Semi-join / correlated subquery rewrite ──────────────────────────────

    #[test]
    fn correlated_exists_returns_matching_rows() {
        let db = Database::in_memory();
        db.sql("CREATE TABLE orders (user_id INT, amount INT)").unwrap();
        db.sql("CREATE TABLE users  (id INT, name TEXT)").unwrap();
        db.sql("INSERT INTO orders VALUES (1, 100), (2, 200)").unwrap();
        db.sql("INSERT INTO users  VALUES (1, 'alice'), (2, 'bob'), (3, 'carol')").unwrap();
        // Users who have placed at least one order.
        let r = db.sql(
            "SELECT name FROM users WHERE EXISTS (SELECT 1 FROM orders WHERE orders.user_id = users.id)"
        ).unwrap();
        assert_eq!(r.row_count(), 2, "alice and bob have orders");
    }

    #[test]
    fn correlated_not_exists_returns_non_matching_rows() {
        let db = Database::in_memory();
        db.sql("CREATE TABLE orders (user_id INT)").unwrap();
        db.sql("CREATE TABLE users  (id INT, name TEXT)").unwrap();
        db.sql("INSERT INTO orders VALUES (1)").unwrap();
        db.sql("INSERT INTO users  VALUES (1, 'alice'), (2, 'bob')").unwrap();
        // Users with NO orders.
        let r = db.sql(
            "SELECT name FROM users WHERE NOT EXISTS (SELECT 1 FROM orders WHERE orders.user_id = users.id)"
        ).unwrap();
        assert_eq!(r.row_count(), 1, "only bob has no orders");
        assert_eq!(r.to_rows()[0][0], "bob");
    }

    // ─── Join reordering (cost-based) ─────────────────────────────────────────

    #[test]
    fn join_reordering_produces_correct_results() {
        let db = Database::in_memory();
        // Create tables of very different sizes.
        db.sql("CREATE TABLE big   (id INT, val INT)").unwrap();
        db.sql("CREATE TABLE small (id INT, label TEXT)").unwrap();
        for i in 1..=100 {
            db.sql(&format!("INSERT INTO big VALUES ({i}, {i})")).unwrap();
        }
        db.sql("INSERT INTO small VALUES (1, 'one'), (2, 'two')").unwrap();
        // Join should work regardless of which table is listed first in FROM.
        let r = db.sql(
            "SELECT label FROM big JOIN small ON big.id = small.id"
        ).unwrap();
        assert_eq!(r.row_count(), 2);
    }
}
