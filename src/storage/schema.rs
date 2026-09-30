use super::types::{DataType, FalconError, FalconResult};

/// Describes a single column: its name and type.
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub name:      String,
    pub data_type: DataType,
    pub nullable:  bool,
}

impl Field {
    pub fn new(name: impl Into<String>, data_type: DataType, nullable: bool) -> Self {
        Self { name: name.into(), data_type, nullable }
    }

    pub fn nullable(name: impl Into<String>, data_type: DataType) -> Self {
        Self::new(name, data_type, true)
    }

    pub fn required(name: impl Into<String>, data_type: DataType) -> Self {
        Self::new(name, data_type, false)
    }
}

/// Describes the full structure of a table.
#[derive(Debug, Clone, PartialEq)]
pub struct Schema {
    pub fields: Vec<Field>,
}

impl Schema {
    pub fn new(fields: Vec<Field>) -> Self {
        Self { fields }
    }

    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Find a column by name, accepting `table.column` as well as `column`.
    ///
    /// The parser turns `ints.i` into the literal string "ints.i", and this
    /// compared names only for exact equality, so every qualified reference
    /// outside a join's `ON` clause failed with "column not found":
    ///
    /// ```text
    ///   SELECT ints.i FROM ints                    -- error
    ///   SELECT i FROM ints WHERE ints.i > 0        -- error
    ///   SELECT a.i FROM ints AS a                  -- error
    ///   SELECT i FROM ints ORDER BY ints.i         -- error
    ///   SELECT i FROM ints JOIN floats ON ints.i = floats.f   -- worked
    /// ```
    ///
    /// The `ON` clause worked because the join resolves its keys itself, which is
    /// why the gap went unnoticed: qualified names had been made to work exactly
    /// where someone needed them.
    ///
    /// Exact match still wins, so a schema that really does hold a field named
    /// "ints.i" -- which a join output can -- keeps its current behaviour.
    ///
    /// The qualifier is checked but cannot always disambiguate: a join output that
    /// keeps unqualified names has no way to tell one table's `i` from another's,
    /// and this reports that as ambiguous rather than silently picking the first.
    /// Guessing there would give wrong rows, which is worse than an error message.
    ///
    /// Split in two so the common case stays small enough to inline. Folding the
    /// qualified-name handling into one function made it large, and the loss
    /// showed up where a query looks columns up most often: the
    /// `WHERE`+`GROUP BY`+`ORDER BY` benchmark at 10k rows went 22.8us -> 26.1us,
    /// confirmed by two runs while the rival engines held still. A structural
    /// argument that a few enum checks could not cost that was wrong; the cost was
    /// not in the new code running, it was in the old code no longer inlining.
    #[inline]
    pub fn column_index(&self, name: &str) -> FalconResult<usize> {
        if let Some(i) = self.fields.iter().position(|f| f.name.eq_ignore_ascii_case(name)) {
            return Ok(i);
        }
        self.column_index_qualified(name)
    }

    /// The `table.column` case, which only runs once the exact match has failed.
    #[cold]
    fn column_index_qualified(&self, name: &str) -> FalconResult<usize> {
        if let Some((qualifier, bare)) = name.rsplit_once('.') {
            // A field carrying the same qualifier is the unambiguous answer.
            if let Some(i) = self
                .fields
                .iter()
                .position(|f| f.name.eq_ignore_ascii_case(name) || {
                    matches!(f.name.rsplit_once('.'), Some((q, b))
                        if q.eq_ignore_ascii_case(qualifier) && b.eq_ignore_ascii_case(bare))
                })
            {
                return Ok(i);
            }

            let mut hits = self
                .fields
                .iter()
                .enumerate()
                .filter(|(_, f)| f.name.eq_ignore_ascii_case(bare));
            if let Some((i, _)) = hits.next() {
                if hits.next().is_none() {
                    return Ok(i);
                }
                return Err(FalconError::InvalidQuery(format!(
                    "column '{bare}' is ambiguous and the qualifier '{qualifier}' cannot                      resolve it, because the schema here keeps unqualified column names"
                )));
            }
        }

        Err(FalconError::ColumnNotFound(name.to_string()))
    }

    pub fn field(&self, name: &str) -> FalconResult<&Field> {
        let idx = self.column_index(name)?;
        Ok(&self.fields[idx])
    }

    pub fn has_column(&self, name: &str) -> bool {
        self.column_index(name).is_ok()
    }

    pub fn column_names(&self) -> Vec<&str> {
        self.fields.iter().map(|f| f.name.as_str()).collect()
    }

    pub fn project(&self, names: &[&str]) -> FalconResult<Schema> {
        let mut projected = Vec::with_capacity(names.len());
        for &name in names {
            let field = self.field(name)?;
            projected.push(field.clone());
        }
        Ok(Schema::new(projected))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_schema() -> Schema {
        Schema::new(vec![
            Field::required("id", DataType::Int64),
            Field::nullable("name", DataType::Utf8),
            Field::nullable("score", DataType::Float64),
        ])
    }

    #[test]
    fn column_index_case_insensitive() {
        let s = sample_schema();
        assert_eq!(s.column_index("NAME").unwrap(), 1);
        assert_eq!(s.column_index("Score").unwrap(), 2);
    }

    #[test]
    fn column_not_found() {
        let s = sample_schema();
        assert!(s.column_index("missing").is_err());
    }

    #[test]
    fn project() {
        let s = sample_schema();
        let projected = s.project(&["id", "score"]).unwrap();
        assert_eq!(projected.field_count(), 2);
        assert_eq!(projected.fields[0].name, "id");
        assert_eq!(projected.fields[1].name, "score");
    }
}
