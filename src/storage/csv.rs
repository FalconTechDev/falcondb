use std::{fs, path::Path};

use rayon::prelude::*;

use super::{
    column::{BooleanColumn, Column, PrimitiveColumn, Utf8Column},
    schema::{Field, Schema},
    table::{RecordBatch, Table},
    types::{parse_date, DataType, FalconError, FalconResult},
};

// ─── Public API ───────────────────────────────────────────────────────────────

/// Load a CSV file into a Table, auto-inferring column types.
pub fn read_csv(path: impl AsRef<Path>, table_name: &str) -> FalconResult<Table> {
    let opts = CsvReadOptions::default();
    CsvReader::new(opts).read(path, table_name)
}

/// Load a CSV with explicit options.
pub fn read_csv_with_options(
    path: impl AsRef<Path>,
    table_name: &str,
    opts: CsvReadOptions,
) -> FalconResult<Table> {
    CsvReader::new(opts).read(path, table_name)
}

// ─── Options ─────────────────────────────────────────────────────────────────

pub struct CsvReadOptions {
    pub delimiter:        u8,
    pub has_header:       bool,
    pub batch_size:       usize,
    pub infer_rows:       usize,
    pub null_values:      Vec<String>,
}

impl Default for CsvReadOptions {
    fn default() -> Self {
        Self {
            delimiter:   b',',
            has_header:  true,
            batch_size:  Table::DEFAULT_BATCH_SIZE,
            infer_rows:  128,
            null_values: vec!["".into(), "NULL".into(), "null".into(), "NA".into(), "N/A".into()],
        }
    }
}

// ─── Reader ──────────────────────────────────────────────────────────────────

struct CsvReader {
    opts: CsvReadOptions,
}

impl CsvReader {
    fn new(opts: CsvReadOptions) -> Self {
        Self { opts }
    }

    fn read(&self, path: impl AsRef<Path>, table_name: &str) -> FalconResult<Table> {
        let path = path.as_ref();
        let content = fs::read_to_string(path)
            .map_err(FalconError::Io)?;

        let mut lines: Vec<&str> = content.lines().collect();
        if lines.is_empty() {
            return Err(FalconError::ParseError("CSV file is empty".into()));
        }

        let headers: Vec<String> = if self.opts.has_header {
            self.parse_row(lines.remove(0))
                .into_iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            (0..self.parse_row(lines[0]).len())
                .map(|i| format!("col{i}"))
                .collect()
        };

        let col_count = headers.len();
        if col_count == 0 {
            return Err(FalconError::ParseError("CSV has no columns".into()));
        }

        // Infer types from the first `infer_rows` data rows.
        let sample_end = self.opts.infer_rows.min(lines.len());
        let types = self.infer_types(&lines[..sample_end], col_count);

        let schema = Schema::new(
            headers.iter().zip(types.iter())
                .map(|(name, dt)| Field::nullable(name.as_str(), dt.clone()))
                .collect::<Vec<_>>(),
        );

        // Parse all rows into raw string cells.
        let raw: Vec<Vec<String>> = lines
            .par_iter()
            .map(|line| self.parse_row(line).into_iter().map(String::from).collect())
            .collect();

        // Split into batches and build columns in parallel per batch.
        let batches: FalconResult<Vec<RecordBatch>> = raw
            .par_chunks(self.opts.batch_size)
            .map(|chunk| self.build_batch(chunk, &schema))
            .collect();

        let mut table = Table::new(table_name, schema);
        for batch in batches? {
            table.append_batch(batch)?;
        }

        Ok(table)
    }

    fn parse_row<'a>(&self, line: &'a str) -> Vec<&'a str> {
        line.split(self.opts.delimiter as char).map(|s| s.trim()).collect()
    }

    fn infer_types(&self, rows: &[&str], col_count: usize) -> Vec<DataType> {
        let mut col_types: Vec<TypeCandidate> = vec![TypeCandidate::Boolean; col_count];
        // `parse_bool` accepts "1"/"0" so that a known-boolean column reads them as
        // true/false. For *inference* that is too eager: a plain integer column whose
        // values happen to be only 0 and 1 would silently become Boolean. Require at
        // least one unambiguous token (true/false/yes/no/t/f) before committing.
        let mut saw_bool_word: Vec<bool> = vec![false; col_count];

        for &row in rows {
            let cells = self.parse_row(row);
            for (i, cell) in cells.iter().enumerate().take(col_count) {
                if i >= col_types.len() { break; }
                if self.is_null(cell) { continue; }
                if is_bool_word(cell) { saw_bool_word[i] = true; }
                col_types[i] = col_types[i].clone().demote(cell);
            }
        }

        col_types
            .into_iter()
            .enumerate()
            .map(|(i, c)| match c {
                TypeCandidate::Boolean if !saw_bool_word[i] => DataType::Int64,
                other => other.into(),
            })
            .collect()
    }

    fn is_null(&self, cell: &str) -> bool {
        self.opts.null_values.iter().any(|n| n == cell)
    }

    fn build_batch(&self, rows: &[Vec<String>], schema: &Schema) -> FalconResult<RecordBatch> {
        let col_count = schema.field_count();
        let row_count = rows.len();

        // Transpose: rows → columns.
        let columns: FalconResult<Vec<Column>> = (0..col_count)
            .into_par_iter()
            .map(|col_idx| {
                let field = &schema.fields[col_idx];
                self.build_column(rows, row_count, col_idx, &field.data_type)
            })
            .collect();

        RecordBatch::new(schema.clone(), columns?)
    }

    fn build_column(
        &self,
        rows:      &[Vec<String>],
        row_count: usize,
        col_idx:   usize,
        dt:        &DataType,
    ) -> FalconResult<Column> {
        Ok(match dt {
            DataType::Boolean => {
                let mut vals = Vec::with_capacity(row_count);
                for row in rows {
                    let cell = row.get(col_idx).map(|s| s.as_str()).unwrap_or("");
                    vals.push(if self.is_null(cell) {
                        None
                    } else {
                        Some(parse_bool(cell)?)
                    });
                }
                Column::Boolean(BooleanColumn::new(vals))
            }
            DataType::Int64 => {
                let mut vals = Vec::with_capacity(row_count);
                for row in rows {
                    let cell = row.get(col_idx).map(|s| s.as_str()).unwrap_or("");
                    vals.push(if self.is_null(cell) {
                        None
                    } else {
                        Some(cell.parse::<i64>().map_err(|_| {
                            FalconError::ParseError(format!("cannot parse '{cell}' as INT64"))
                        })?)
                    });
                }
                Column::Int64(PrimitiveColumn::new(vals))
            }
            DataType::Float64 => {
                let mut vals = Vec::with_capacity(row_count);
                for row in rows {
                    let cell = row.get(col_idx).map(|s| s.as_str()).unwrap_or("");
                    vals.push(if self.is_null(cell) {
                        None
                    } else {
                        Some(cell.parse::<f64>().map_err(|_| {
                            FalconError::ParseError(format!("cannot parse '{cell}' as FLOAT64"))
                        })?)
                    });
                }
                Column::Float64(PrimitiveColumn::new(vals))
            }
            DataType::Date32 => {
                let mut vals = Vec::with_capacity(row_count);
                for row in rows {
                    let cell = row.get(col_idx).map(|s| s.as_str()).unwrap_or("");
                    vals.push(if self.is_null(cell) {
                        None
                    } else {
                        Some(parse_date(cell).ok_or_else(|| {
                            FalconError::ParseError(format!("cannot parse '{cell}' as DATE"))
                        })?)
                    });
                }
                Column::Date32(PrimitiveColumn::new(vals))
            }
            _ => {
                let vals: Vec<Option<String>> = rows.iter().map(|row| {
                    let cell = row.get(col_idx).map(|s| s.as_str()).unwrap_or("");
                    if self.is_null(cell) { None } else { Some(cell.to_string()) }
                }).collect();
                Column::Utf8(Utf8Column::new(vals))
            }
        })
    }
}

/// Is this cell an unambiguous boolean token? Digits deliberately do not count —
/// see `infer_types`.
fn is_bool_word(cell: &str) -> bool {
    matches!(
        cell.to_ascii_lowercase().as_str(),
        "true" | "false" | "yes" | "no" | "t" | "f"
    )
}

// ─── Type inference ───────────────────────────────────────────────────────────

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
#[allow(clippy::enum_variant_names)]
enum TypeCandidate {
    Boolean,
    Int64,
    Float64,
    Date32,
    Utf8,
}

impl TypeCandidate {
    /// Demote the candidate if `cell` cannot be parsed as the current type.
    fn demote(self, cell: &str) -> Self {
        match self {
            TypeCandidate::Boolean => {
                if parse_bool(cell).is_ok() { TypeCandidate::Boolean }
                else { TypeCandidate::Int64.demote(cell) }
            }
            TypeCandidate::Int64 => {
                if cell.parse::<i64>().is_ok() { TypeCandidate::Int64 }
                else { TypeCandidate::Float64.demote(cell) }
            }
            TypeCandidate::Float64 => {
                if cell.parse::<f64>().is_ok() { TypeCandidate::Float64 }
                else { TypeCandidate::Date32.demote(cell) }
            }
            TypeCandidate::Date32 => {
                if parse_date(cell).is_some() { TypeCandidate::Date32 }
                else { TypeCandidate::Utf8 }
            }
            TypeCandidate::Utf8 => TypeCandidate::Utf8,
        }
    }
}

impl From<TypeCandidate> for DataType {
    fn from(c: TypeCandidate) -> Self {
        match c {
            TypeCandidate::Boolean => DataType::Boolean,
            TypeCandidate::Int64   => DataType::Int64,
            TypeCandidate::Float64 => DataType::Float64,
            TypeCandidate::Date32  => DataType::Date32,
            TypeCandidate::Utf8    => DataType::Utf8,
        }
    }
}

fn parse_bool(cell: &str) -> FalconResult<bool> {
    match cell.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "t" => Ok(true),
        "false" | "0" | "no" | "f" => Ok(false),
        _ => Err(FalconError::ParseError(format!("cannot parse '{cell}' as BOOLEAN"))),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn csv_table(content: &str) -> Table {
        let mut f = NamedTempFile::new().unwrap();
        write!(f, "{content}").unwrap();
        read_csv(f.path(), "test").unwrap()
    }

    #[test]
    fn basic_load() {
        let t = csv_table("id,name,score\n1,alice,9.5\n2,bob,8.0\n3,carol,7.2\n");
        assert_eq!(t.row_count(), 3);
        assert_eq!(t.schema.field_count(), 3);
    }

    #[test]
    fn type_inference() {
        let t = csv_table("a,b,c\n1,3.14,hello\n2,2.72,world\n");
        assert_eq!(t.schema.fields[0].data_type, DataType::Int64);
        assert_eq!(t.schema.fields[1].data_type, DataType::Float64);
        assert_eq!(t.schema.fields[2].data_type, DataType::Utf8);
    }

    #[test]
    fn null_handling() {
        let t = csv_table("id,val\n1,10\n2,NULL\n3,\n");
        let batch = &t.batches[0];
        let col = batch.column_by_name("val").unwrap();
        assert_eq!(col.get(0).unwrap(), super::super::types::ScalarValue::Int64(Some(10)));
        assert_eq!(col.get(1).unwrap(), super::super::types::ScalarValue::Int64(None));
        assert_eq!(col.get(2).unwrap(), super::super::types::ScalarValue::Int64(None));
    }

    #[test]
    fn boolean_inference() {
        let t = csv_table("flag\ntrue\nfalse\n1\n0\n");
        assert_eq!(t.schema.fields[0].data_type, DataType::Boolean);
    }
}
