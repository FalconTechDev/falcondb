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

    pub fn column_index(&self, name: &str) -> FalconResult<usize> {
        self.fields
            .iter()
            .position(|f| f.name.eq_ignore_ascii_case(name))
            .ok_or_else(|| FalconError::ColumnNotFound(name.to_string()))
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
