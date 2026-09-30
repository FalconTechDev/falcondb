use sqlparser::{
    ast::{
        BinaryOperator, Expr as SqlExpr, FunctionArguments, FunctionArg,
        GroupByExpr, JoinConstraint, JoinOperator, Offset as SqlOffset, OrderByExpr, Query,
        SelectItem, SetExpr, Statement, TableFactor, UnaryOperator, Value,
    },
    dialect::GenericDialect,
    parser::Parser,
};

use crate::storage::types::{FalconError, FalconResult, ScalarValue};

use super::{
    aggregate::{AggExpr, AggFunc, GroupByExec},
    dml::{DeleteStmt, InsertStmt, UpdateStmt},
    expr::{BinaryOp, Expr},
    select::{SelectExpr, SelectQuery},
    sort::{SortExpr, SortOrder},
    window::{WindowExpr, WindowFrame, WindowFunc},
};

// в”Ђв”Ђв”Ђ Public entry point в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

#[derive(Clone, Debug)]
pub struct ParsedQuery {
    pub table:    String,
    /// `FROM emp e` в†’ `Some("e")`. Needed to tell an outer reference (`e.dept`)
    /// apart from an inner column that happens to share the name.
    pub alias:    Option<String>,
    pub select:   SelectQuery,
    pub group_by: Option<GroupByExec>,
    pub having:   Option<Expr>,
    pub sort:     Vec<SortExpr>,
    pub joins:    Vec<JoinClause>,
    /// Set operations chained onto this query, in written order:
    /// `a UNION b INTERSECT c` → `[(UnionDistinct, b), (Intersect, c)]`.
    /// Each branch is a full query and runs through the whole pipeline.
    pub set_ops:  Vec<(SetOp, ParsedQuery)>,
    pub windows:  Vec<WindowExpr>,
    /// `WITH name AS (вЂ¦)` bindings, in declaration order. Each is materialised into a
    /// temporary table before the main query runs, so a later CTE can reference an
    /// earlier one. The `bool` is `true` for RECURSIVE CTEs.
    pub ctes:     Vec<(String, ParsedQuery, bool)>,
}

/// Source format for COPY ... FROM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopyFormat {
    Csv,
    Parquet,
}

/// A SQL set operator joining two query blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SetOp {
    /// `UNION ALL` — concatenate, keep duplicates.
    UnionAll,
    /// `UNION` — concatenate, then remove duplicate rows.
    UnionDistinct,
    /// `INTERSECT` — rows present in both sides, deduplicated.
    Intersect,
    /// `EXCEPT` — rows on the left that do not appear on the right, deduplicated.
    Except,
}

impl std::fmt::Display for SetOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            SetOp::UnionAll      => "UNION ALL",
            SetOp::UnionDistinct => "UNION",
            SetOp::Intersect     => "INTERSECT",
            SetOp::Except        => "EXCEPT",
        })
    }
}

#[derive(Clone, Debug)]
pub struct JoinClause {
    pub table:     String,
    pub on:        Expr,
    pub join_type: crate::query::join::JoinType,
}

pub fn parse_sql(sql: &str) -> FalconResult<ParsedQuery> {
    match parse_statement(sql)? {
        ParsedStatement::Query(q) => Ok(q),
        _ => Err(FalconError::InvalidQuery(
            "expected a SELECT statement".into(),
        )),
    }
}

/// Parse a bare SQL expression (e.g. the body of a CHECK constraint).
pub fn parse_check_expr(sql_expr: &str) -> FalconResult<super::expr::Expr> {
    let wrapped = format!("SELECT * FROM _dummy_ WHERE {sql_expr}");
    let dialect = sqlparser::dialect::GenericDialect {};
    let mut stmts = sqlparser::parser::Parser::parse_sql(&dialect, &wrapped)
        .map_err(|e| FalconError::ParseError(e.to_string()))?;
    if let Some(Statement::Query(q)) = stmts.pop() {
        if let SetExpr::Select(sel) = *q.body {
            if let Some(filter) = sel.selection {
                return translate_expr(&filter);
            }
        }
    }
    Err(FalconError::ParseError(format!("could not parse CHECK expression: {sql_expr}")))
}

// в”Ђв”Ђв”Ђ DDL types в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

/// Column definition for CREATE TABLE / ALTER TABLE ADD COLUMN.
#[derive(Debug, Clone)]
pub struct DdlColumnDef {
    pub name:      String,
    pub data_type: crate::storage::types::DataType,
    pub nullable:  bool,
}

/// CREATE TABLE statement.
#[derive(Debug, Clone)]
pub struct CreateTableStmt {
    pub name:          String,
    pub columns:       Vec<DdlColumnDef>,
    pub if_not_exists: bool,
    pub constraints:   Vec<crate::storage::table::TableConstraint>,
}

/// Operations supported by ALTER TABLE.
#[derive(Debug, Clone)]
pub enum AlterAction {
    AddColumn(DdlColumnDef),
    DropColumn(String),
    AddConstraint(crate::storage::table::TableConstraint),
    DropConstraint(String),
}

/// ALTER TABLE statement.
#[derive(Debug, Clone)]
pub struct AlterTableStmt {
    pub table:  String,
    pub action: AlterAction,
}

/// CREATE INDEX statement.
#[derive(Debug, Clone)]
pub struct CreateIndexStmt {
    pub name:          String,
    pub table:         String,
    pub column:        String,
    pub index_type:    crate::storage::table::IndexType,
    pub unique:        bool,
    pub if_not_exists: bool,
}

// в”Ђв”Ђв”Ђ Statement dispatch в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

/// Any statement FalconDB can execute.
#[derive(Clone)]
pub enum ParsedStatement {
    Query(ParsedQuery),
    Insert(InsertStmt),
    Update(UpdateStmt),
    Delete(DeleteStmt),
    CreateTable(CreateTableStmt),
    DropTable { name: String, if_exists: bool },
    AlterTable(AlterTableStmt),
    /// `EXPLAIN stmt`, or `EXPLAIN ANALYZE stmt` when `analyze` is set. ANALYZE
    /// runs the statement and reports what actually happened; plain EXPLAIN only
    /// describes the plan and touches no data.
    Explain { stmt: Box<ParsedStatement>, analyze: bool },
    Begin,
    Commit,
    Rollback,
    Truncate { name: String },
    /// Flush all tables to `.fdb` files in `path` and truncate the WAL.
    Checkpoint { path: Option<String> },
    /// `VACUUM [table]` — compact fragmented batches. Without a name, every table.
    Vacuum { table: Option<String> },
    /// `COPY <table> FROM '<path>' [(FORMAT CSV|PARQUET)]` — bulk-load a file into
    /// a table, replacing whatever it held.
    Copy { table: String, path: String, format: CopyFormat },
    CreateIndex(CreateIndexStmt),
    DropIndex { name: String, if_exists: bool },
}

pub fn parse_statement(sql: &str) -> FalconResult<ParsedStatement> {
    // CHECKPOINT is not standard SQL вЂ” handle it before the parser.
    let upper = sql.trim().to_ascii_uppercase();
    if upper == "CHECKPOINT" || upper.starts_with("CHECKPOINT ") || upper.starts_with("CHECKPOINT\t") {
        let rest = sql.trim()["CHECKPOINT".len()..].trim();
        let path = if rest.is_empty() {
            None
        } else {
            Some(rest.trim_matches('\'').to_string())
        };
        return Ok(ParsedStatement::Checkpoint { path });
    }

    if upper == "VACUUM" || upper.starts_with("VACUUM ") || upper.starts_with("VACUUM	") {
        let rest = sql.trim()["VACUUM".len()..].trim().trim_end_matches(';').trim();
        let table = if rest.is_empty() { None } else { Some(rest.to_string()) };
        return Ok(ParsedStatement::Vacuum { table });
    }

    if upper.starts_with("COPY ") {
        return parse_copy(sql.trim());
    }

    let dialect = GenericDialect {};
    let mut stmts = Parser::parse_sql(&dialect, sql)
        .map_err(|e| FalconError::ParseError(e.to_string()))?;

    if stmts.len() != 1 {
        return Err(FalconError::InvalidQuery(
            "exactly one statement expected".into(),
        ));
    }

    translate_statement(stmts.remove(0))
}

/// Parse `COPY <table> FROM '<path>' [(FORMAT CSV|PARQUET)]`.
///
/// Hand-written rather than delegated to sqlparser, whose COPY node models the
/// PostgreSQL wire-protocol form (`COPY ... FROM STDIN`) instead of the
/// load-a-file form that is useful here.
///
/// The format defaults to the file extension, so the common case needs no
/// FORMAT clause. Only FROM is supported; COPY ... TO is not implemented, and
/// says so rather than being read as FROM.
fn parse_copy(sql: &str) -> FalconResult<ParsedStatement> {
    let body = sql.trim().trim_end_matches(';').trim();
    let rest = &body["COPY".len()..];

    // Split on the FROM keyword, case-insensitively.
    let upper = rest.to_ascii_uppercase();
    let from_at = upper.find(" FROM ").ok_or_else(|| {
        if upper.contains(" TO ") {
            FalconError::InvalidQuery("COPY ... TO is not supported; use COPY ... FROM".into())
        } else {
            FalconError::InvalidQuery("COPY requires FROM '<path>'".into())
        }
    })?;

    let table = rest[..from_at].trim().to_string();
    if table.is_empty() {
        return Err(FalconError::InvalidQuery("COPY requires a table name".into()));
    }

    let after_from = rest[from_at + " FROM ".len()..].trim();

    // The path is a single-quoted literal; anything after it is the option list.
    let mut chars = after_from.char_indices();
    if !matches!(chars.next(), Some((_, '\''))) {
        return Err(FalconError::InvalidQuery(
            "COPY path must be a single-quoted string".into(),
        ));
    }
    let close = after_from[1..].find('\'').ok_or_else(|| {
        FalconError::InvalidQuery("unterminated path string in COPY".into())
    })? + 1;
    let path = after_from[1..close].to_string();

    let opts = after_from[close + 1..].trim().to_ascii_uppercase();
    let format = if opts.contains("PARQUET") {
        CopyFormat::Parquet
    } else if opts.contains("CSV") {
        CopyFormat::Csv
    } else if path.to_ascii_lowercase().ends_with(".parquet") {
        CopyFormat::Parquet
    } else {
        CopyFormat::Csv
    };

    Ok(ParsedStatement::Copy { table, path, format })
}

fn translate_statement(stmt: Statement) -> FalconResult<ParsedStatement> {
    match stmt {
        Statement::Query(q) => Ok(ParsedStatement::Query(translate_query(*q)?)),
        Statement::Insert(ins) => Ok(ParsedStatement::Insert(translate_insert(ins)?)),
        Statement::Update { table, assignments, selection, returning, .. } => {
            Ok(ParsedStatement::Update(translate_update(table, assignments, selection, returning)?))
        }
        Statement::Delete(del) => Ok(ParsedStatement::Delete(translate_delete(del)?)),
        Statement::CreateTable { name, columns, if_not_exists, constraints, .. } => {
            let table_name = name.to_string();
            let mut table_constraints: Vec<crate::storage::table::TableConstraint> = Vec::new();

            let ddl_columns = columns.iter()
                .map(|col| {
                    use sqlparser::ast::ColumnOption;
                    let col_name = col.name.value.clone();
                    let dt = translate_data_type(&col.data_type)?;
                    let mut nullable = true;
                    for opt in &col.options {
                        match &opt.option {
                            ColumnOption::NotNull => { nullable = false; }
                            ColumnOption::Unique { is_primary, .. } => {
                                let cname = opt.name.as_ref().map(|n| n.value.clone());
                                let kind = if *is_primary {
                                    nullable = false;
                                    crate::storage::table::ConstraintKind::PrimaryKey(
                                        vec![col_name.clone()]
                                    )
                                } else {
                                    crate::storage::table::ConstraintKind::Unique(
                                        vec![col_name.clone()]
                                    )
                                };
                                table_constraints.push(crate::storage::table::TableConstraint {
                                    name: cname,
                                    kind,
                                });
                            }
                            ColumnOption::Check(expr) => {
                                table_constraints.push(crate::storage::table::TableConstraint {
                                    name: opt.name.as_ref().map(|n| n.value.clone()),
                                    kind: crate::storage::table::ConstraintKind::Check(
                                        expr.to_string()
                                    ),
                                });
                            }
                            _ => {}
                        }
                    }
                    Ok(DdlColumnDef { name: col_name, data_type: dt, nullable })
                })
                .collect::<FalconResult<Vec<_>>>()?;

            // Table-level constraints
            for tc in &constraints {
                use sqlparser::ast::TableConstraint;
                match tc {
                    TableConstraint::PrimaryKey { name, columns, .. } => {
                        let cols: Vec<String> = columns.iter()
                            .map(|c| c.value.clone())
                            .collect();
                        // Mark those columns NOT NULL in ddl_columns
                        table_constraints.push(crate::storage::table::TableConstraint {
                            name: name.as_ref().map(|n| n.value.clone()),
                            kind: crate::storage::table::ConstraintKind::PrimaryKey(cols),
                        });
                    }
                    TableConstraint::Unique { name, columns, .. } => {
                        let cols: Vec<String> = columns.iter()
                            .map(|c| c.value.clone())
                            .collect();
                        table_constraints.push(crate::storage::table::TableConstraint {
                            name: name.as_ref().map(|n| n.value.clone()),
                            kind: crate::storage::table::ConstraintKind::Unique(cols),
                        });
                    }
                    TableConstraint::Check { name, expr } => {
                        table_constraints.push(crate::storage::table::TableConstraint {
                            name: name.as_ref().map(|n| n.value.clone()),
                            kind: crate::storage::table::ConstraintKind::Check(
                                expr.to_string()
                            ),
                        });
                    }
                    TableConstraint::ForeignKey {
                        name,
                        columns,
                        foreign_table,
                        referred_columns,
                        ..
                    } => {
                        let cols: Vec<String> = columns.iter().map(|c| c.value.clone()).collect();
                        let ref_cols: Vec<String> = referred_columns.iter().map(|c| c.value.clone()).collect();
                        table_constraints.push(crate::storage::table::TableConstraint {
                            name: name.as_ref().map(|n| n.value.clone()),
                            kind: crate::storage::table::ConstraintKind::ForeignKey {
                                columns:     cols,
                                ref_table:   foreign_table.to_string(),
                                ref_columns: ref_cols,
                            },
                        });
                    }
                    _ => {} // Index, etc. — not supported
                }
            }

            Ok(ParsedStatement::CreateTable(CreateTableStmt {
                name: table_name,
                columns: ddl_columns,
                if_not_exists,
                constraints: table_constraints,
            }))
        }
        Statement::CreateIndex { name, table_name, columns, unique, if_not_exists, using, .. } => {
            let idx_name = name
                .as_ref()
                .map(|n| n.to_string())
                .unwrap_or_else(|| "_unnamed_".to_string());
            let table = table_name.to_string();
            let column = columns.first()
                .ok_or_else(|| FalconError::InvalidQuery(
                    "CREATE INDEX requires at least one column".into()
                ))?
                .expr
                .to_string();
            let index_type = match using.as_ref().map(|id| id.value.to_uppercase()).as_deref() {
                Some("HASH") => crate::storage::table::IndexType::Hash,
                _ => crate::storage::table::IndexType::BTree,
            };
            Ok(ParsedStatement::CreateIndex(CreateIndexStmt {
                name: idx_name,
                table,
                column,
                index_type,
                unique,
                if_not_exists,
            }))
        }
        Statement::Drop { object_type, names, if_exists, .. } => {
            use sqlparser::ast::ObjectType;
            match object_type {
                ObjectType::Table => {
                    let name = names.first()
                        .map(|n| n.to_string())
                        .ok_or_else(|| FalconError::InvalidQuery(
                            "DROP TABLE requires a table name".into()
                        ))?;
                    Ok(ParsedStatement::DropTable { name, if_exists })
                }
                ObjectType::Index => {
                    let name = names.first()
                        .map(|n| n.to_string())
                        .ok_or_else(|| FalconError::InvalidQuery(
                            "DROP INDEX requires an index name".into()
                        ))?;
                    Ok(ParsedStatement::DropIndex { name, if_exists })
                }
                other => Err(FalconError::InvalidQuery(
                    format!("unsupported DROP target: {other:?}")
                )),
            }
        }
        Statement::AlterTable { name, operations, .. } => {
            translate_alter_table(name.to_string(), operations)
        }
        Statement::Explain { statement, analyze, .. } => {
            let inner = translate_statement(*statement)?;
            Ok(ParsedStatement::Explain { stmt: Box::new(inner), analyze })
        }
        Statement::Truncate { table_name, .. } => {
            Ok(ParsedStatement::Truncate { name: table_name.to_string() })
        }
        Statement::StartTransaction { .. } => Ok(ParsedStatement::Begin),
        Statement::Commit { .. }           => Ok(ParsedStatement::Commit),
        Statement::Rollback { .. }         => Ok(ParsedStatement::Rollback),
        other => Err(FalconError::InvalidQuery(format!(
            "unsupported statement: {other}"
        ))),
    }
}

// в”Ђв”Ђв”Ђ INSERT в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

fn translate_insert(ins: sqlparser::ast::Insert) -> FalconResult<InsertStmt> {
    use crate::query::dml::InsertSource;

    let table = ins.table_name.to_string();
    let columns: Vec<String> = ins.columns.iter().map(|c| c.value.clone()).collect();

    let query = ins.source.ok_or_else(|| {
        FalconError::InvalidQuery("INSERT requires a VALUES or SELECT clause".into())
    })?;

    let source = if matches!(*query.body, SetExpr::Values(_)) {
        let SetExpr::Values(values) = *query.body else { unreachable!() };
        let rows = values
            .rows
            .iter()
            .map(|row| row.iter().map(translate_expr).collect::<FalconResult<Vec<_>>>())
            .collect::<FalconResult<Vec<_>>>()?;
        InsertSource::Values(rows)
    } else {
        // INSERT вЂ¦ SELECT вЂ” translate the full Query (preserves ORDER BY / LIMIT).
        InsertSource::Select(Box::new(translate_query(*query)?))
    };

    let returning = translate_returning(ins.returning.as_deref())?;
    Ok(InsertStmt { table, columns, source, returning })
}

// в”Ђв”Ђв”Ђ UPDATE в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

fn translate_update(
    table:       sqlparser::ast::TableWithJoins,
    assignments: Vec<sqlparser::ast::Assignment>,
    selection:   Option<SqlExpr>,
    returning:   Option<Vec<sqlparser::ast::SelectItem>>,
) -> FalconResult<UpdateStmt> {
    let name = match &table.relation {
        TableFactor::Table { name, .. } => name.to_string(),
        other => {
            return Err(FalconError::InvalidQuery(format!(
                "unsupported UPDATE target: {other}"
            )))
        }
    };

    let assignments = assignments
        .iter()
        .map(|a| {
            let col = assignment_column(a)?;
            Ok((col, translate_expr(&a.value)?))
        })
        .collect::<FalconResult<Vec<_>>>()?;

    let filter = selection.as_ref().map(translate_expr).transpose()?;
    let returning = translate_returning(returning.as_deref())?;

    Ok(UpdateStmt { table: name, assignments, filter, returning })
}

fn assignment_column(a: &sqlparser::ast::Assignment) -> FalconResult<String> {
    // `id` is a column path (`t.col`); the last segment is the column name.
    a.id
        .last()
        .map(|p| p.value.clone())
        .ok_or_else(|| FalconError::InvalidQuery("empty assignment target".into()))
}

// в”Ђв”Ђв”Ђ DELETE в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

fn translate_delete(del: sqlparser::ast::Delete) -> FalconResult<DeleteStmt> {
    use sqlparser::ast::FromTable;

    let tables = match &del.from {
        FromTable::WithFromKeyword(t) | FromTable::WithoutKeyword(t) => t,
    };

    let first = tables.first().ok_or_else(|| {
        FalconError::InvalidQuery("DELETE requires a target table".into())
    })?;

    let name = match &first.relation {
        TableFactor::Table { name, .. } => name.to_string(),
        other => {
            return Err(FalconError::InvalidQuery(format!(
                "unsupported DELETE target: {other}"
            )))
        }
    };

    let filter = del.selection.as_ref().map(translate_expr).transpose()?;
    let returning = translate_returning(del.returning.as_deref())?;

    Ok(DeleteStmt { table: name, filter, returning })
}

/// Translate a `RETURNING` clause (a slice of `SelectItem`) into `SelectExpr` list.
fn translate_returning(
    items: Option<&[sqlparser::ast::SelectItem]>,
) -> FalconResult<Vec<SelectExpr>> {
    let Some(items) = items else { return Ok(vec![]) };
    items.iter().enumerate().map(|(i, item)| {
        match item {
            SelectItem::UnnamedExpr(e) => {
                let expr = translate_expr(e)?;
                let alias = expr_alias(e, i);
                Ok(SelectExpr { expr, alias })
            }
            SelectItem::ExprWithAlias { expr: e, alias } => {
                let expr = translate_expr(e)?;
                Ok(SelectExpr { expr, alias: alias.value.clone() })
            }
            SelectItem::Wildcard(_) => {
                // RETURNING * — sentinel; executor expands to all columns.
                Ok(SelectExpr { expr: Expr::Column("*".into()), alias: "*".into() })
            }
            SelectItem::QualifiedWildcard(name, _) => {
                Err(FalconError::InvalidQuery(format!(
                    "qualified wildcard {name}.* is not supported in RETURNING"
                )))
            }
        }
    }).collect()
}

// в”Ђв”Ђв”Ђ Query translation в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

fn translate_query(query: Query) -> FalconResult<ParsedQuery> {
    // в”Ђв”Ђ WITH вЂ¦ AS (вЂ¦) в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
    let ctes = match &query.with {
        Some(with) => {
            let recursive = with.recursive;
            with.cte_tables
                .iter()
                .map(|cte| {
                    let name = cte.alias.name.value.clone();
                    let plan = translate_query((*cte.query).clone())?;
                    Ok((name, plan, recursive))
                })
                .collect::<FalconResult<Vec<_>>>()?
        }
        None => Vec::new(),
    };

    let limit = match &query.limit {
        Some(SqlExpr::Value(Value::Number(n, _))) => {
            Some(n.parse::<usize>().map_err(|_| {
                FalconError::ParseError(format!("invalid LIMIT value: {n}"))
            })?)
        }
        None => None,
        _ => return Err(FalconError::InvalidQuery("LIMIT must be a literal integer".into())),
    };

    let offset = match &query.offset {
        Some(SqlOffset { value: SqlExpr::Value(Value::Number(n, _)), .. }) => {
            n.parse::<usize>().map_err(|_| {
                FalconError::ParseError(format!("invalid OFFSET value: {n}"))
            })?
        }
        None => 0,
        _ => return Err(FalconError::InvalidQuery("OFFSET must be a literal integer".into())),
    };

    let sort = query.order_by.iter()
        .map(translate_order_by)
        .collect::<FalconResult<Vec<_>>>()?;

    // Handle UNION ALL by flattening into a list of sub-queries.
    let (body, set_ops) = flatten_unions(*query.body)?;
    let SetExpr::Select(sel) = body else {
        return Err(FalconError::InvalidQuery("only plain SELECT is supported".into()));
    };

    let distinct = matches!(sel.distinct, Some(_));

    // в”Ђв”Ђ FROM + JOINs в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
    let from_item = sel.from.first()
        .ok_or_else(|| FalconError::InvalidQuery("FROM clause is required".into()))?;

    let (table, alias) = match &from_item.relation {
        TableFactor::Table { name, alias, .. } => (
            name.to_string(),
            alias.as_ref().map(|a| a.name.value.clone()),
        ),
        other => return Err(FalconError::InvalidQuery(
            format!("unsupported FROM target: {other}")
        )),
    };

    let mut joins = Vec::new();
    for join in &from_item.joins {
        let join_table = match &join.relation {
            TableFactor::Table { name, .. } => name.to_string(),
            other => return Err(FalconError::InvalidQuery(
                format!("unsupported JOIN target: {other}")
            )),
        };
        use crate::query::join::JoinType;
        use crate::query::expr::Expr as FExpr;
        let (on, join_type) = match &join.join_operator {
            JoinOperator::Inner(JoinConstraint::On(expr))         => (translate_expr(expr)?, JoinType::Inner),
            JoinOperator::LeftOuter(JoinConstraint::On(expr))     => (translate_expr(expr)?, JoinType::Left),
            JoinOperator::RightOuter(JoinConstraint::On(expr))    => (translate_expr(expr)?, JoinType::Right),
            JoinOperator::FullOuter(JoinConstraint::On(expr))     => (translate_expr(expr)?, JoinType::Full),
            // CROSS JOIN has no ON clause — store a TRUE placeholder; executor routes via JoinType::Cross.
            JoinOperator::CrossJoin => (FExpr::Literal(crate::storage::types::ScalarValue::Boolean(Some(true))), JoinType::Cross),
            // NATURAL JOIN: treated as INNER with the cross-product; caller must deduplicate.
            JoinOperator::Inner(JoinConstraint::Natural)
            | JoinOperator::LeftOuter(JoinConstraint::Natural)
            | JoinOperator::RightOuter(JoinConstraint::Natural) => {
                return Err(FalconError::InvalidQuery(
                    "NATURAL JOIN is not supported — use an explicit ON condition".into()
                ))
            }
            other => return Err(FalconError::InvalidQuery(
                format!("unsupported JOIN type: {other:?}")
            )),
        };
        joins.push(JoinClause { table: join_table, on, join_type });
    }

    let filter  = sel.selection.as_ref().map(translate_expr).transpose()?;

    let (projections, mut agg_exprs, group_by_exprs, windows) =
        classify_projection(&sel.projection, &sel.group_by)?;

    // HAVING is translated after the aggregates are known, because it runs
    // against the *aggregated* batch, whose columns are the output aliases.
    //
    // Translating it like any other expression meant only the alias form worked:
    //
    //     HAVING c > 10          ok
    //     HAVING COUNT(*) > 10   "unsupported function arg"
    //     HAVING COUNT(n) > 10   "column 'n' not found"
    //     HAVING SUM(n) > 100    same
    //
    // -- so the textbook spelling of the clause failed while an alias, which not
    // every dialect even allows there, was the only thing that worked. Each
    // aggregate in the clause now becomes a reference to the column that
    // aggregate produces, and one the SELECT list does not already compute is
    // added to the group-by so the reference has something to point at.
    let having = sel.having.as_ref()
        .map(|h| translate_having(h, &mut agg_exprs))
        .transpose()?;

    let group_by = if !agg_exprs.is_empty() || !group_by_exprs.is_empty() {
        Some(GroupByExec::new(group_by_exprs, agg_exprs))
    } else {
        None
    };

    let mut select = SelectQuery::scan_all();
    if let Some(f) = filter { select = select.with_filter(f); }
    if let Some(n) = limit  { select = select.with_limit(n); }
    if offset > 0            { select = select.with_offset(offset); }
    if distinct              { select = select.with_distinct(true); }
    // Set explicit column list when we have projections AND:
    // - no GROUP BY (GROUP BY handles its own output columns), or
    // - there are window functions that need the alias list to survive projection.
    if !projections.is_empty() && (group_by.is_none() || !windows.is_empty()) {
        select = select.with_columns(projections);
    }

    Ok(ParsedQuery { table, alias, select, group_by, having, sort, joins, set_ops, windows, ctes })
}

// в”Ђв”Ђв”Ђ Projection classification в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

fn classify_projection(
    items:    &[SelectItem],
    group_by: &GroupByExpr,
) -> FalconResult<(Vec<SelectExpr>, Vec<AggExpr>, Vec<Expr>, Vec<WindowExpr>)> {
    let mut projections  = Vec::new();
    let mut agg_exprs    = Vec::new();
    let mut window_exprs = Vec::new();

    for (i, item) in items.iter().enumerate() {
        match item {
            SelectItem::Wildcard(_) => {}

            SelectItem::UnnamedExpr(e) => {
                let alias = expr_alias(e, i);
                if let Some(win) = try_translate_window(e, &alias)? {
                    // Window column will be appended at execution time вЂ” project by alias.
                    projections.push(SelectExpr::named(Expr::Column(alias.clone()), &alias));
                    window_exprs.push(win);
                } else if let Some(agg) = try_translate_agg(e, &alias)? {
                    agg_exprs.push(agg);
                } else {
                    projections.push(SelectExpr::named(translate_expr(e)?, &alias));
                }
            }

            SelectItem::ExprWithAlias { expr: e, alias } => {
                let al = alias.value.clone();
                if let Some(win) = try_translate_window(e, &al)? {
                    projections.push(SelectExpr::named(Expr::Column(al.clone()), &al));
                    window_exprs.push(win);
                } else if let Some(agg) = try_translate_agg(e, &al)? {
                    agg_exprs.push(agg);
                } else {
                    projections.push(SelectExpr::named(translate_expr(e)?, &al));
                }
            }

            SelectItem::QualifiedWildcard(_, _) => {}
        }
    }

    let gb_exprs = match group_by {
        GroupByExpr::Expressions(exprs) => exprs
            .iter()
            .map(translate_expr)
            .collect::<FalconResult<Vec<_>>>()?,
        GroupByExpr::All => Vec::new(),
    };

    Ok((projections, agg_exprs, gb_exprs, window_exprs))
}

fn expr_alias(e: &SqlExpr, idx: usize) -> String {
    match e {
        SqlExpr::Identifier(id)    => id.value.clone(),
        SqlExpr::CompoundIdentifier(parts) => parts.last()
            .map(|p| p.value.clone())
            .unwrap_or_else(|| format!("col{idx}")),
        _ => format!("col{idx}"),
    }
}

// в”Ђв”Ђв”Ђ Aggregate detection в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

/// Translate a `HAVING` clause, turning each aggregate into a reference to the
/// column that aggregate produces.
///
/// `HAVING` is evaluated against the aggregated batch, so `COUNT(*)` there cannot
/// be computed -- the rows it would count are gone. What it can do is name the
/// column holding the count. When the SELECT list already computes that
/// aggregate, its alias is used; when it does not, the aggregate is appended with
/// a generated name so the reference resolves. The generated name is prefixed to
/// keep it clear of anything a user would write.
fn translate_having(e: &SqlExpr, aggs: &mut Vec<AggExpr>) -> FalconResult<Expr> {
    // An aggregate here is replaced wholesale by the column it produces.
    if let Some(agg) = try_translate_agg(e, "")? {
        let existing = aggs.iter().find(|a| {
            format!("{:?}", a.func) == format!("{:?}", agg.func)
                && a.input.to_string() == agg.input.to_string()
        });
        let name = match existing {
            Some(a) => a.alias.clone(),
            None => {
                let name = format!("__having_{}", aggs.len());
                aggs.push(AggExpr { alias: name.clone(), ..agg });
                name
            }
        };
        return Ok(Expr::Column(name));
    }

    // Otherwise recurse through the shapes a HAVING clause is built from, so an
    // aggregate nested inside a comparison or a conjunction is still found.
    // Anything else is translated as it always was.
    Ok(match e {
        SqlExpr::BinaryOp { left, op, right } => Expr::BinaryOp {
            left:  Box::new(translate_having(left, aggs)?),
            op:    translate_binary_op(op)?,
            right: Box::new(translate_having(right, aggs)?),
        },
        SqlExpr::Nested(inner) => translate_having(inner, aggs)?,
        SqlExpr::UnaryOp { op: sqlparser::ast::UnaryOperator::Not, expr } => {
            Expr::Not(Box::new(translate_having(expr, aggs)?))
        }
        SqlExpr::IsNull(inner)    => Expr::IsNull(Box::new(translate_having(inner, aggs)?)),
        SqlExpr::IsNotNull(inner) => Expr::IsNotNull(Box::new(translate_having(inner, aggs)?)),
        other => translate_expr(other)?,
    })
}

fn try_translate_agg(e: &SqlExpr, alias: &str) -> FalconResult<Option<AggExpr>> {
    let SqlExpr::Function(func) = e else { return Ok(None); };

    let name = func.name.to_string().to_uppercase();

    // Detect COUNT(DISTINCT col)
    let is_distinct = matches!(&func.args,
        FunctionArguments::List(l) if l.duplicate_treatment == Some(sqlparser::ast::DuplicateTreatment::Distinct)
    );

    let agg_func = match name.as_str() {
        "COUNT" if is_distinct => AggFunc::CountDistinct,
        "COUNT" => AggFunc::Count,
        "SUM"   => AggFunc::Sum,
        "AVG"   => AggFunc::Avg,
        "MIN"   => AggFunc::Min,
        "MAX"   => AggFunc::Max,
        "STRING_AGG" | "GROUP_CONCAT" | "LISTAGG" => {
            // Second arg is the separator string
            let sep = match &func.args {
                FunctionArguments::List(list) => {
                    list.args.get(1).and_then(|a| match a {
                        FunctionArg::Unnamed(sqlparser::ast::FunctionArgExpr::Expr(
                            SqlExpr::Value(Value::SingleQuotedString(s))
                        )) => Some(s.clone()),
                        _ => None,
                    }).unwrap_or_else(|| ",".to_string())
                }
                _ => ",".to_string(),
            };
            AggFunc::StringAgg { separator: sep }
        }
        "BOOL_AND" | "EVERY" => AggFunc::BoolAnd,
        "BOOL_OR"  | "SOME"  => AggFunc::BoolOr,
        "STDDEV" | "STDDEV_SAMP" | "STD" => AggFunc::Stddev { population: false },
        "STDDEV_POP"  => AggFunc::Stddev   { population: true  },
        "VARIANCE" | "VAR_SAMP" | "VAR" => AggFunc::Variance { population: false },
        "VAR_POP"     => AggFunc::Variance { population: true  },
        "FIRST"  | "FIRST_VALUE_AGG" => AggFunc::First,
        "LAST"   | "LAST_VALUE_AGG"  => AggFunc::Last,
        _       => return Ok(None),
    };

    let input = match &func.args {
        FunctionArguments::List(list) => {
            match list.args.first() {
                Some(FunctionArg::Unnamed(sqlparser::ast::FunctionArgExpr::Wildcard)) => {
                    Expr::Literal(ScalarValue::Int64(Some(1)))
                }
                Some(FunctionArg::Unnamed(sqlparser::ast::FunctionArgExpr::Expr(e))) => {
                    translate_expr(e)?
                }
                _ => Expr::Literal(ScalarValue::Int64(Some(1))),
            }
        }
        _ => Expr::Literal(ScalarValue::Int64(Some(1))),
    };

    Ok(Some(AggExpr { func: agg_func, input, alias: alias.to_string() }))
}

// в”Ђв”Ђв”Ђ Window function detection в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

fn try_translate_window(e: &SqlExpr, alias: &str) -> FalconResult<Option<WindowExpr>> {
    let SqlExpr::Function(func) = e else { return Ok(None); };

    // sqlparser represents `func(...) OVER (...)` by setting `func.over`.
    let over = match &func.over {
        Some(o) => o,
        None => return Ok(None),
    };

    let name = func.name.to_string().to_uppercase();

    let window_func = match name.as_str() {
        "ROW_NUMBER"  => WindowFunc::RowNumber,
        "RANK"        => WindowFunc::Rank,
        "DENSE_RANK"  => WindowFunc::DenseRank,
        "LAG"  => {
            let offset = extract_window_offset(&func.args, 1)?;
            WindowFunc::Lag(offset)
        }
        "LEAD" => {
            let offset = extract_window_offset(&func.args, 1)?;
            WindowFunc::Lead(offset)
        }
        "SUM"         => WindowFunc::Sum,
        "AVG"         => WindowFunc::Avg,
        "MIN"         => WindowFunc::Min,
        "MAX"         => WindowFunc::Max,
        "COUNT"       => WindowFunc::Count,
        "FIRST_VALUE" => WindowFunc::FirstValue,
        "LAST_VALUE"  => WindowFunc::LastValue,
        "NTH_VALUE"   => {
            let n = extract_window_offset(&func.args, 2)?;
            WindowFunc::NthValue(n.max(1))
        }
        "PERCENT_RANK" => WindowFunc::PercentRank,
        "CUME_DIST"    => WindowFunc::CumeDist,
        "NTILE"        => {
            let n = extract_window_offset(&func.args, 1)?;
            WindowFunc::Ntile(n.max(1))
        }
        _       => return Ok(None),
    };

    // Input expression (the first arg for SUM/AVG/MIN/MAX/LAG/LEAD).
    let input = match name.as_str() {
        "ROW_NUMBER" | "RANK" | "DENSE_RANK" | "COUNT"
        | "PERCENT_RANK" | "CUME_DIST" | "NTILE" => None,
        _ => {
            match &func.args {
                FunctionArguments::List(list) => {
                    list.args.first().and_then(|a| match a {
                        FunctionArg::Unnamed(sqlparser::ast::FunctionArgExpr::Expr(e)) =>
                            translate_expr(e).ok().map(Some),
                        _ => None,
                    }).flatten()
                }
                _ => None,
            }
        }
    };

    // PARTITION BY.
    let partition_by = match over {
        sqlparser::ast::WindowType::WindowSpec(spec) => spec.partition_by.iter()
            .map(translate_expr)
            .collect::<FalconResult<Vec<_>>>()?,
        sqlparser::ast::WindowType::NamedWindow(_) => vec![],
    };

    // ORDER BY inside the window.
    let order_by = match over {
        sqlparser::ast::WindowType::WindowSpec(spec) => spec.order_by.iter()
            .map(translate_order_by)
            .collect::<FalconResult<Vec<_>>>()?,
        sqlparser::ast::WindowType::NamedWindow(_) => vec![],
    };

    // Frame: `ROWS BETWEEN n PRECEDING AND m FOLLOWING`, defaulting to the SQL
    // standard cumulative frame when the query does not spell one out.
    let frame = match over {
        sqlparser::ast::WindowType::WindowSpec(spec) => translate_window_frame(&spec.window_frame)?,
        sqlparser::ast::WindowType::NamedWindow(_)   => WindowFrame::Cumulative,
    };

    Ok(Some(WindowExpr {
        func: window_func,
        input,
        partition_by,
        order_by,
        frame,
        alias: alias.to_string(),
    }))
}

/// Translate a window frame clause.
///
/// Only ROWS frames are supported. A RANGE frame is bounded by *value* distance
/// rather than row count, so silently treating it as ROWS would return wrong
/// numbers on any data with duplicate order keys — it is rejected instead.
fn translate_window_frame(
    frame: &Option<sqlparser::ast::WindowFrame>,
) -> FalconResult<WindowFrame> {
    use sqlparser::ast::{WindowFrameBound, WindowFrameUnits};

    let Some(f) = frame else { return Ok(WindowFrame::Cumulative) };

    if !matches!(f.units, WindowFrameUnits::Rows) {
        return Err(FalconError::InvalidQuery(format!(
            "only ROWS window frames are supported, got {:?}",
            f.units
        )));
    }

    // A bound offset must be a non-negative integer literal.
    let offset = |b: &WindowFrameBound| -> FalconResult<Option<usize>> {
        let expr = match b {
            WindowFrameBound::CurrentRow => return Ok(Some(0)),
            // Unbounded on either side: report it as "no limit" for the caller to map.
            WindowFrameBound::Preceding(None) | WindowFrameBound::Following(None) => {
                return Ok(None)
            }
            WindowFrameBound::Preceding(Some(e)) | WindowFrameBound::Following(Some(e)) => e,
        };
        match translate_expr(expr)? {
            Expr::Literal(ScalarValue::Int64(Some(n))) if n >= 0 => Ok(Some(n as usize)),
            other => Err(FalconError::InvalidQuery(format!(
                "window frame offset must be a non-negative integer, got {other}"
            ))),
        }
    };

    let start = offset(&f.start_bound)?;
    let end = match &f.end_bound {
        Some(b) => offset(b)?,
        // `ROWS n PRECEDING` with no BETWEEN ends at the current row.
        None => Some(0),
    };

    Ok(match (start, end) {
        // UNBOUNDED PRECEDING AND CURRENT ROW — standard cumulative frame.
        (None, Some(0)) => WindowFrame::Cumulative,
        // UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING — whole partition at every row.
        (None, None)    => WindowFrame::WholePartition,
        // N PRECEDING AND UNBOUNDED FOLLOWING (incl. CURRENT ROW = 0 PRECEDING).
        (Some(preceding), None) => WindowFrame::UnboundedFollowing { preceding },
        // Bounded sliding window.
        (Some(preceding), Some(following)) => WindowFrame::Rows { preceding, following },
        // UNBOUNDED PRECEDING AND N FOLLOWING — unusual but valid; treat as bounded
        // from the start of the partition (preceding = large sentinel).
        (None, Some(following)) => WindowFrame::Rows { preceding: usize::MAX / 2, following },
    })
}

fn extract_window_offset(args: &FunctionArguments, default: usize) -> FalconResult<usize> {
    match args {
        FunctionArguments::List(list) if list.args.len() >= 2 => {
            match &list.args[1] {
                FunctionArg::Unnamed(sqlparser::ast::FunctionArgExpr::Expr(
                    SqlExpr::Value(Value::Number(n, _))
                )) => n.parse::<usize>().map_err(|_|
                    FalconError::ParseError(format!("invalid LAG/LEAD offset: {n}"))
                ),
                _ => Ok(default),
            }
        }
        _ => Ok(default),
    }
}

// в”Ђв”Ђв”Ђ Expression translation в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

fn translate_expr(e: &SqlExpr) -> FalconResult<Expr> {
    match e {
        SqlExpr::Identifier(id) => Ok(Expr::Column(id.value.clone())),

        SqlExpr::CompoundIdentifier(parts) => {
            let name = parts.iter()
                .map(|p| p.value.clone())
                .collect::<Vec<_>>()
                .join(".");
            Ok(Expr::Column(name))
        }

        SqlExpr::Value(v) => Ok(Expr::Literal(translate_value(v)?)),

        // Scalar subquery: `(SELECT AVG(x) FROM t)`.
        SqlExpr::Subquery(q) => Ok(Expr::Subquery(crate::query::expr::SubqueryPlan(
            Box::new(translate_query((**q).clone())?),
        ))),

        // `x IN (SELECT вЂ¦)` / `x NOT IN (SELECT вЂ¦)`.
        SqlExpr::InSubquery { expr, subquery, negated } => Ok(Expr::InSubquery {
            expr:    Box::new(translate_expr(expr)?),
            plan:    crate::query::expr::SubqueryPlan(Box::new(translate_query((**subquery).clone())?)),
            negated: *negated,
        }),

        // `EXISTS (SELECT вЂ¦)` / `NOT EXISTS (вЂ¦)`.
        SqlExpr::Exists { subquery, negated } => Ok(Expr::Exists {
            plan:    crate::query::expr::SubqueryPlan(Box::new(translate_query((**subquery).clone())?)),
            negated: *negated,
        }),

        SqlExpr::BinaryOp { left, op, right } => {
            let l = translate_expr(left)?;
            let r = translate_expr(right)?;
            let o = translate_binary_op(op)?;
            Ok(Expr::BinaryOp { left: Box::new(l), op: o, right: Box::new(r) })
        }

        SqlExpr::UnaryOp { op: UnaryOperator::Not, expr } => {
            Ok(Expr::Not(Box::new(translate_expr(expr)?)))
        }

        SqlExpr::UnaryOp { op: UnaryOperator::Minus, expr } => {
            // Fold -(literal) into a negative literal when possible; otherwise represent
            // as (0 - expr) so eval_batch handles it through the normal BinaryOp path.
            let inner = translate_expr(expr)?;
            if let Expr::Literal(ref sv) = inner {
                use crate::storage::types::ScalarValue;
                let negated = match sv {
                    ScalarValue::Int32(Some(n))  => Some(Expr::Literal(ScalarValue::Int32(Some(-n)))),
                    ScalarValue::Int64(Some(n))  => Some(Expr::Literal(ScalarValue::Int64(Some(-n)))),
                    ScalarValue::Float32(Some(f)) => Some(Expr::Literal(ScalarValue::Float32(Some(-f)))),
                    ScalarValue::Float64(Some(f)) => Some(Expr::Literal(ScalarValue::Float64(Some(-f)))),
                    _ => None,
                };
                if let Some(neg) = negated {
                    return Ok(neg);
                }
            }
            Ok(Expr::BinaryOp {
                left:  Box::new(Expr::Literal(crate::storage::types::ScalarValue::Int64(Some(0)))),
                op:    BinaryOp::Sub,
                right: Box::new(inner),
            })
        }

        SqlExpr::UnaryOp { op: UnaryOperator::Plus, expr } => {
            translate_expr(expr)
        }

        SqlExpr::IsNull(inner) => {
            Ok(Expr::IsNull(Box::new(translate_expr(inner)?)))
        }

        SqlExpr::IsNotNull(inner) => {
            Ok(Expr::IsNotNull(Box::new(translate_expr(inner)?)))
        }

        SqlExpr::Nested(inner) => translate_expr(inner),

        SqlExpr::Between { expr, negated, low, high } => {
            let col  = translate_expr(expr)?;
            let lo   = translate_expr(low)?;
            let hi   = translate_expr(high)?;
            let lo_cmp = Expr::BinaryOp {
                left:  Box::new(col.clone()),
                op:    BinaryOp::GtEq,
                right: Box::new(lo),
            };
            let hi_cmp = Expr::BinaryOp {
                left:  Box::new(col),
                op:    BinaryOp::LtEq,
                right: Box::new(hi),
            };
            let between = Expr::BinaryOp {
                left:  Box::new(lo_cmp),
                op:    BinaryOp::And,
                right: Box::new(hi_cmp),
            };
            if *negated { Ok(Expr::Not(Box::new(between))) } else { Ok(between) }
        }

        // LIKE / ILIKE
        SqlExpr::Like { expr, pattern, negated, .. } => {
            let inner   = translate_expr(expr)?;
            let pat_str = extract_string_literal(pattern)?;
            Ok(Expr::Like {
                expr:             Box::new(inner),
                pattern:          pat_str,
                negated:          *negated,
                case_insensitive: false,
            })
        }
        SqlExpr::ILike { expr, pattern, negated, .. } => {
            let inner   = translate_expr(expr)?;
            let pat_str = extract_string_literal(pattern)?;
            Ok(Expr::Like {
                expr:             Box::new(inner),
                pattern:          pat_str,
                negated:          *negated,
                case_insensitive: true,
            })
        }

        // IN (v1, v2, ...)
        SqlExpr::InList { expr, list, negated } => {
            let inner = translate_expr(expr)?;
            let items = list.iter().map(translate_expr).collect::<FalconResult<_>>()?;
            Ok(Expr::InList { expr: Box::new(inner), list: items, negated: *negated })
        }

        // CASE WHEN c1 THEN r1 вЂ¦ ELSE e END
        SqlExpr::Case { conditions, results, else_result, .. } => {
            let when_thens = conditions.iter().zip(results.iter())
                .map(|(c, r)| Ok((translate_expr(c)?, translate_expr(r)?)))
                .collect::<FalconResult<Vec<_>>>()?;
            let else_expr = else_result.as_ref()
                .map(|e| translate_expr(e).map(Box::new))
                .transpose()?;
            Ok(Expr::Case { when_thens, else_expr })
        }

        // Scalar function calls (non-aggregate)
        SqlExpr::Function(func) => {
            let name = func.name.to_string();
            let args = match &func.args {
                FunctionArguments::List(list) => list.args.iter()
                    .filter_map(|a| match a {
                        FunctionArg::Unnamed(sqlparser::ast::FunctionArgExpr::Expr(e)) => {
                            Some(translate_expr(e))
                        }
                        // `COUNT(*)` — the star is not an expression, and rejecting
                        // it here made `HAVING COUNT(*) > 10` fail outright with
                        // "unsupported function arg". It carries no information a
                        // scalar call needs, so it is dropped rather than refused.
                        FunctionArg::Unnamed(sqlparser::ast::FunctionArgExpr::Wildcard) => None,
                        _ => Some(Err(FalconError::InvalidQuery("unsupported function arg".into()))),
                    })
                    .collect::<FalconResult<Vec<_>>>()?,
                _ => vec![],
            };
            Ok(Expr::Function { name, args })
        }

        // CAST(expr AS type)
        SqlExpr::Cast { expr, data_type, .. } => Ok(Expr::Cast {
            expr:    Box::new(translate_expr(expr)?),
            to_type: translate_cast_type(data_type)?,
        }),

        // EXTRACT(field FROM expr)
        SqlExpr::Extract { field, expr, .. } => {
            use sqlparser::ast::DateTimeField as SqlField;
            use crate::query::expr::DateTimeField;
            let f = match field {
                SqlField::Year    => DateTimeField::Year,
                SqlField::Month   => DateTimeField::Month,
                SqlField::Day     => DateTimeField::Day,
                SqlField::Hour    => DateTimeField::Hour,
                SqlField::Minute  => DateTimeField::Minute,
                SqlField::Second  => DateTimeField::Second,
                SqlField::Epoch   => DateTimeField::Epoch,
                other => return Err(FalconError::InvalidQuery(
                    format!("unsupported EXTRACT field: {other:?}")
                )),
            };
            Ok(Expr::Extract { field: f, expr: Box::new(translate_expr(expr)?) })
        }

        // TRIM, SUBSTRING, CEIL and FLOOR are parsed into their own AST nodes
        // rather than as ordinary function calls, so they never reached the
        // `Function` arm above and fell through to the error below. All four are
        // implemented in `eval_function` and were unreachable from SQL:
        //
        //     LTRIM and RTRIM worked, TRIM did not.
        //     SUBSTR worked, SUBSTRING did not.
        //     ROUND worked, CEIL and FLOOR did not.
        //
        // Translated to the same `Expr::Function` the evaluator already answers,
        // so nothing new is being implemented here -- only connected.
        SqlExpr::Trim { expr, .. } => Ok(Expr::Function {
            name: "TRIM".into(),
            args: vec![translate_expr(expr)?],
        }),

        SqlExpr::Substring { expr, substring_from, substring_for, .. } => {
            let mut args = vec![translate_expr(expr)?];
            if let Some(from) = substring_from { args.push(translate_expr(from)?); }
            if let Some(len)  = substring_for  { args.push(translate_expr(len)?); }
            Ok(Expr::Function { name: "SUBSTR".into(), args })
        }

        SqlExpr::Ceil { expr, .. } => Ok(Expr::Function {
            name: "CEIL".into(),
            args: vec![translate_expr(expr)?],
        }),

        SqlExpr::Floor { expr, .. } => Ok(Expr::Function {
            name: "FLOOR".into(),
            args: vec![translate_expr(expr)?],
        }),

        _ => Err(FalconError::InvalidQuery(
            format!("unsupported expression: {e}")
        )),
    }
}

/// Flatten a chain of set operations into (leftmost_body, [(op, branch), ...]).
///
/// SQL set operators are left-associative, so `a UNION b EXCEPT c` parses as
/// `(a UNION b) EXCEPT c`. Recursing down the left spine and pushing each right
/// branch afterwards restores written order, which is the order the executor has
/// to apply them in: EXCEPT and INTERSECT do not commute.
fn flatten_unions(body: SetExpr) -> FalconResult<(SetExpr, Vec<(SetOp, ParsedQuery)>)> {
    use sqlparser::ast::{SetOperator, SetQuantifier};
    match body {
        SetExpr::SetOperation { op, set_quantifier, left, right } => {
            let all = matches!(set_quantifier, SetQuantifier::All | SetQuantifier::ByName);
            let set_op = match op {
                SetOperator::Union if all => SetOp::UnionAll,
                SetOperator::Union        => SetOp::UnionDistinct,
                SetOperator::Intersect    => SetOp::Intersect,
                SetOperator::Except       => SetOp::Except,
            };
            let (base, mut parts) = flatten_unions(*left)?;
            // Translate the right side as a full sub-query.
            let right_q = translate_query(sqlparser::ast::Query {
                with:      None,
                body:      right,
                order_by:  vec![],
                limit:     None,
                offset:    None,
                fetch:     None,
                locks:     vec![],
                limit_by:  vec![],
                for_clause: None,
            })?;
            parts.push((set_op, right_q));
            Ok((base, parts))
        }
        other => Ok((other, vec![])),
    }
}

// в”Ђв”Ђв”Ђ DDL translation в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

fn translate_alter_table(
    table:      String,
    operations: Vec<sqlparser::ast::AlterTableOperation>,
) -> FalconResult<ParsedStatement> {
    use sqlparser::ast::AlterTableOperation;
    let op = operations.into_iter().next()
        .ok_or_else(|| FalconError::InvalidQuery(
            "ALTER TABLE requires an operation".into()
        ))?;

    let action = match op {
        AlterTableOperation::AddColumn { column_def, .. } => {
            let col_name = column_def.name.value.clone();
            let dt = translate_data_type(&column_def.data_type)?;
            let nullable = !column_def.options.iter().any(|o| {
                matches!(o.option, sqlparser::ast::ColumnOption::NotNull)
            });
            AlterAction::AddColumn(DdlColumnDef { name: col_name, data_type: dt, nullable })
        }
        AlterTableOperation::DropColumn { column_name, .. } => {
            AlterAction::DropColumn(column_name.value.clone())
        }
        AlterTableOperation::AddConstraint(tc) => {
            use sqlparser::ast::TableConstraint;
            let constraint = match tc {
                TableConstraint::PrimaryKey { name, columns, .. } => {
                    crate::storage::table::TableConstraint {
                        name: name.as_ref().map(|n| n.value.clone()),
                        kind: crate::storage::table::ConstraintKind::PrimaryKey(
                            columns.iter().map(|c| c.value.clone()).collect()
                        ),
                    }
                }
                TableConstraint::Unique { name, columns, .. } => {
                    crate::storage::table::TableConstraint {
                        name: name.as_ref().map(|n| n.value.clone()),
                        kind: crate::storage::table::ConstraintKind::Unique(
                            columns.iter().map(|c| c.value.clone()).collect()
                        ),
                    }
                }
                TableConstraint::Check { name, expr } => {
                    crate::storage::table::TableConstraint {
                        name: name.as_ref().map(|n| n.value.clone()),
                        kind: crate::storage::table::ConstraintKind::Check(expr.to_string()),
                    }
                }
                other => return Err(FalconError::InvalidQuery(
                    format!("unsupported ADD CONSTRAINT type: {other:?}")
                )),
            };
            AlterAction::AddConstraint(constraint)
        }
        AlterTableOperation::DropConstraint { name, .. } => {
            AlterAction::DropConstraint(name.value.clone())
        }
        other => return Err(FalconError::InvalidQuery(
            format!("unsupported ALTER TABLE operation: {other:?}")
        )),
    };

    Ok(ParsedStatement::AlterTable(AlterTableStmt { table, action }))
}

fn translate_data_type(dt: &sqlparser::ast::DataType) -> FalconResult<crate::storage::types::DataType> {
    use sqlparser::ast::DataType as SqlDt;
    use crate::storage::types::DataType;
    Ok(match dt {
        SqlDt::Int(_) | SqlDt::Integer(_) | SqlDt::BigInt(_) | SqlDt::SmallInt(_) => DataType::Int64,
        SqlDt::Float(_) | SqlDt::Double | SqlDt::Real => DataType::Float64,
        SqlDt::Varchar(_) | SqlDt::Text | SqlDt::Char(_) | SqlDt::String(_) => DataType::Utf8,
        SqlDt::Boolean => DataType::Boolean,
        SqlDt::Date => DataType::Date32,
        SqlDt::Timestamp(_, _) => DataType::Timestamp,
        SqlDt::Decimal(info) | SqlDt::Numeric(info) => {
            use sqlparser::ast::ExactNumberInfo;
            match info {
                ExactNumberInfo::PrecisionAndScale(p, s) => DataType::Decimal128(*p as u8, *s as i8),
                ExactNumberInfo::Precision(p) => DataType::Decimal128(*p as u8, 0),
                ExactNumberInfo::None => DataType::Decimal128(18, 6),
            }
        }
        other => return Err(FalconError::InvalidQuery(
            format!("unsupported data type: {other}")
        )),
    })
}

fn translate_cast_type(dt: &sqlparser::ast::DataType) -> FalconResult<crate::storage::types::DataType> {
    translate_data_type(dt)
}

fn extract_string_literal(e: &SqlExpr) -> FalconResult<String> {
    match e {
        SqlExpr::Value(Value::SingleQuotedString(s)) |
        SqlExpr::Value(Value::DoubleQuotedString(s)) => Ok(s.clone()),
        other => Err(FalconError::InvalidQuery(
            format!("LIKE pattern must be a string literal, got: {other}")
        )),
    }
}

fn translate_value(v: &Value) -> FalconResult<ScalarValue> {
    match v {
        Value::Number(n, _) => {
            if let Ok(i) = n.parse::<i64>() {
                return Ok(ScalarValue::Int64(Some(i)));
            }
            if let Ok(f) = n.parse::<f64>() {
                return Ok(ScalarValue::Float64(Some(f)));
            }
            Err(FalconError::ParseError(format!("cannot parse number: {n}")))
        }
        Value::SingleQuotedString(s) | Value::DoubleQuotedString(s) => {
            Ok(ScalarValue::Utf8(Some(s.clone())))
        }
        Value::Boolean(b) => Ok(ScalarValue::Boolean(Some(*b))),
        Value::Null        => Ok(ScalarValue::Null),
        _ => Err(FalconError::ParseError(format!("unsupported literal: {v}"))),
    }
}

fn translate_binary_op(op: &BinaryOperator) -> FalconResult<BinaryOp> {
    match op {
        BinaryOperator::Plus    => Ok(BinaryOp::Add),
        BinaryOperator::Minus   => Ok(BinaryOp::Sub),
        BinaryOperator::Multiply => Ok(BinaryOp::Mul),
        BinaryOperator::Divide  => Ok(BinaryOp::Div),
        BinaryOperator::Eq      => Ok(BinaryOp::Eq),
        BinaryOperator::NotEq   => Ok(BinaryOp::NotEq),
        BinaryOperator::Lt      => Ok(BinaryOp::Lt),
        BinaryOperator::LtEq    => Ok(BinaryOp::LtEq),
        BinaryOperator::Gt      => Ok(BinaryOp::Gt),
        BinaryOperator::GtEq    => Ok(BinaryOp::GtEq),
        BinaryOperator::And     => Ok(BinaryOp::And),
        BinaryOperator::Or      => Ok(BinaryOp::Or),
        _ => Err(FalconError::InvalidQuery(format!("unsupported operator: {op}"))),
    }
}

fn translate_order_by(o: &OrderByExpr) -> FalconResult<SortExpr> {
    let expr  = translate_expr(&o.expr)?;
    let order = match o.asc {
        Some(false) => SortOrder::Desc,
        _           => SortOrder::Asc,
    };
    let nulls_last = match (o.nulls_first, &order) {
        (Some(true), _)  => false,
        (Some(false), _) => true,
        (None, SortOrder::Asc)  => false,
        (None, SortOrder::Desc) => true,
    };
    Ok(SortExpr { expr, order, nulls_last })
}

// в”Ђв”Ђв”Ђ Tests в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_select_star() {
        let q = parse_sql("SELECT * FROM users").unwrap();
        assert_eq!(q.table, "users");
        assert!(q.group_by.is_none());
        assert!(q.sort.is_empty());
    }

    #[test]
    fn parse_select_with_where() {
        let q = parse_sql("SELECT name FROM users WHERE score > 8.0").unwrap();
        assert_eq!(q.table, "users");
        assert!(q.select.filter.is_some());
    }

    #[test]
    fn parse_limit_offset() {
        let q = parse_sql("SELECT * FROM t LIMIT 10 OFFSET 5").unwrap();
        assert_eq!(q.select.limit, Some(10));
        assert_eq!(q.select.offset, 5);
    }

    #[test]
    fn parse_order_by() {
        let q = parse_sql("SELECT * FROM t ORDER BY score DESC").unwrap();
        assert_eq!(q.sort.len(), 1);
        assert!(matches!(q.sort[0].order, SortOrder::Desc));
    }

    #[test]
    fn parse_group_by_count() {
        let q = parse_sql("SELECT dept, COUNT(*) as cnt FROM employees GROUP BY dept").unwrap();
        assert!(q.group_by.is_some());
        let gb = q.group_by.unwrap();
        assert_eq!(gb.aggs.len(), 1);
    }

    #[test]
    fn parse_between() {
        let q = parse_sql("SELECT * FROM t WHERE age BETWEEN 20 AND 40").unwrap();
        assert!(q.select.filter.is_some());
    }

    #[test]
    fn parse_is_null() {
        let q = parse_sql("SELECT * FROM t WHERE name IS NULL").unwrap();
        assert!(q.select.filter.is_some());
    }

    #[test]
    fn reject_non_select() {
        assert!(parse_sql("INSERT INTO t VALUES (1)").is_err());
    }
}
