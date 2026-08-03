use std::fmt;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    Boolean,
    Int32,
    Int64,
    Float32,
    Float64,
    Utf8,
    Null,
    /// Days since 1970-01-01.
    Date32,
    /// Microseconds since 1970-01-01 00:00:00 UTC.
    Timestamp,
    /// Fixed-point decimal: value = integer / 10^scale.
    Decimal128(u8, i8),  // (precision, scale)
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataType::Boolean          => write!(f, "BOOLEAN"),
            DataType::Int32            => write!(f, "INT32"),
            DataType::Int64            => write!(f, "INT64"),
            DataType::Float32          => write!(f, "FLOAT32"),
            DataType::Float64          => write!(f, "FLOAT64"),
            DataType::Utf8             => write!(f, "VARCHAR"),
            DataType::Null             => write!(f, "NULL"),
            DataType::Date32           => write!(f, "DATE"),
            DataType::Timestamp        => write!(f, "TIMESTAMP"),
            DataType::Decimal128(p, s) => write!(f, "DECIMAL({p},{s})"),
        }
    }
}

/// A single scalar value — used in expressions, filters, results.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum ScalarValue {
    Boolean(Option<bool>),
    Int32(Option<i32>),
    Int64(Option<i64>),
    Float32(Option<f32>),
    Float64(Option<f64>),
    Utf8(Option<String>),
    Date32(Option<i32>),
    Timestamp(Option<i64>),
    /// Boxed so the i128's 16-byte alignment does not set the size of every
    /// ScalarValue. Inline it and the enum is 48 bytes; boxed it is 32, and the
    /// common variants are the ones that get carried around in bulk.
    Decimal128(Box<DecimalValue>),
    Null,
}

/// Payload of `ScalarValue::Decimal128`.
///
/// `PartialOrd` compares the raw integer without reconciling scale, matching how
/// `ScalarValue` already ordered decimals when the fields were inline. Comparing
/// decimals of different scales goes through `as_f64` instead.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct DecimalValue {
    pub value:     Option<i128>,
    pub precision: u8,
    pub scale:     i8,
}

impl ScalarValue {
    /// Build a decimal value without the caller naming the box.
    pub fn decimal(value: Option<i128>, precision: u8, scale: i8) -> Self {
        ScalarValue::Decimal128(Box::new(DecimalValue { value, precision, scale }))
    }

    pub fn data_type(&self) -> DataType {
        match self {
            ScalarValue::Boolean(_)           => DataType::Boolean,
            ScalarValue::Int32(_)             => DataType::Int32,
            ScalarValue::Int64(_)             => DataType::Int64,
            ScalarValue::Float32(_)           => DataType::Float32,
            ScalarValue::Float64(_)           => DataType::Float64,
            ScalarValue::Utf8(_)              => DataType::Utf8,
            ScalarValue::Date32(_)            => DataType::Date32,
            ScalarValue::Timestamp(_)         => DataType::Timestamp,
            ScalarValue::Decimal128(d)        => DataType::Decimal128(d.precision, d.scale),
            ScalarValue::Null                 => DataType::Null,
        }
    }

    pub fn is_null(&self) -> bool {
        match self {
            ScalarValue::Boolean(v)          => v.is_none(),
            ScalarValue::Int32(v)            => v.is_none(),
            ScalarValue::Int64(v)            => v.is_none(),
            ScalarValue::Float32(v)          => v.is_none(),
            ScalarValue::Float64(v)          => v.is_none(),
            ScalarValue::Utf8(v)             => v.is_none(),
            ScalarValue::Date32(v)           => v.is_none(),
            ScalarValue::Timestamp(v)        => v.is_none(),
            ScalarValue::Decimal128(d) => d.value.is_none(),
            ScalarValue::Null                => true,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ScalarValue::Boolean(v) => *v,
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ScalarValue::Int32(v)   => v.map(|n| n as f64),
            ScalarValue::Int64(v)   => v.map(|n| n as f64),
            ScalarValue::Float32(v) => v.map(|n| n as f64),
            ScalarValue::Float64(v) => *v,
            ScalarValue::Decimal128(d) => d.value.map(|n| n as f64 / 10f64.powi(d.scale as i32)),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            ScalarValue::Int32(v) => v.map(|n| n as i64),
            ScalarValue::Int64(v) => *v,
            _ => None,
        }
    }

    pub fn as_date32(&self) -> Option<i32> {
        match self {
            ScalarValue::Date32(v) => *v,
            _ => None,
        }
    }

    pub fn as_timestamp(&self) -> Option<i64> {
        match self {
            ScalarValue::Timestamp(v) => *v,
            _ => None,
        }
    }
}

impl fmt::Display for ScalarValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScalarValue::Boolean(Some(v))          => write!(f, "{v}"),
            ScalarValue::Int32(Some(v))            => write!(f, "{v}"),
            ScalarValue::Int64(Some(v))            => write!(f, "{v}"),
            ScalarValue::Float32(Some(v))          => write!(f, "{v}"),
            ScalarValue::Float64(Some(v))          => write!(f, "{v}"),
            ScalarValue::Utf8(Some(v))             => write!(f, "{v}"),
            ScalarValue::Date32(Some(d))           => write!(f, "{}", date32_to_string(*d)),
            ScalarValue::Timestamp(Some(ts))       => write!(f, "{}", timestamp_to_string(*ts)),
            ScalarValue::Decimal128(d) if d.value.is_some() =>
                write!(f, "{}", decimal_to_string(d.value.unwrap(), d.scale)),
            _ => write!(f, "NULL"),
        }
    }
}

// ─── Date/time helpers ────────────────────────────────────────────────────────

/// Convert days since 1970-01-01 to (year, month, day).
/// Uses Howard Hinnant's civil-from-days algorithm.
pub fn days_to_ymd(z: i32) -> (i32, u32, u32) {
    let z = z as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as i32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y   = yoe + (era * 400) as i32;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp  = (5 * doy + 2) / 153;
    let d   = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m   = if mp < 10 { (mp + 3) as u32 } else { (mp - 9) as u32 };
    let y   = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Convert (year, month, day) to days since 1970-01-01.
pub fn ymd_to_days(y: i32, m: u32, d: u32) -> i32 {
    let y = if m <= 2 { y - 1 } else { y };
    let m = m as i32;
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d as i32 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

pub fn date32_to_string(days: i32) -> String {
    let (y, m, d) = days_to_ymd(days);
    format!("{y:04}-{m:02}-{d:02}")
}

pub fn timestamp_to_string(micros: i64) -> String {
    let total_secs = micros.div_euclid(1_000_000);
    let us_part    = micros.rem_euclid(1_000_000);
    let days       = total_secs.div_euclid(86_400) as i32;
    let secs_day   = total_secs.rem_euclid(86_400);
    let h = secs_day / 3600;
    let m = (secs_day % 3600) / 60;
    let s = secs_day % 60;
    let (yr, mo, d) = days_to_ymd(days);
    if us_part == 0 {
        format!("{yr:04}-{mo:02}-{d:02} {h:02}:{m:02}:{s:02}")
    } else {
        format!("{yr:04}-{mo:02}-{d:02} {h:02}:{m:02}:{s:02}.{us_part:06}")
    }
}

fn decimal_to_string(value: i128, scale: i8) -> String {
    if scale <= 0 {
        let factor = 10i128.pow((-scale) as u32);
        return (value * factor).to_string();
    }
    let factor = 10i128.pow(scale as u32);
    let int_part  = value.abs() / factor;
    let frac_part = value.abs() % factor;
    let sign = if value < 0 { "-" } else { "" };
    format!("{sign}{int_part}.{frac_part:0>width$}", width = scale as usize)
}

/// Parse `YYYY-MM-DD` → days since epoch. Returns None on invalid input.
pub fn parse_date(s: &str) -> Option<i32> {
    let s = s.trim();
    let parts: Vec<&str> = s.splitn(3, '-').collect();
    if parts.len() != 3 { return None; }
    let y: i32 = parts[0].parse().ok()?;
    let m: u32 = parts[1].parse().ok()?;
    let d: u32 = parts[2].parse().ok()?;
    if m < 1 || m > 12 || d < 1 || d > 31 { return None; }
    Some(ymd_to_days(y, m, d))
}

/// Parse `YYYY-MM-DD [T ]HH:MM:SS[.ffffff]` → microseconds since epoch.
pub fn parse_timestamp(s: &str) -> Option<i64> {
    let s = s.trim();
    // Split on T or space between date and time
    let (date_part, time_part) = if let Some(p) = s.find('T').or_else(|| s.find(' ')) {
        (&s[..p], Some(&s[p+1..]))
    } else {
        (s, None)
    };
    let days = parse_date(date_part)? as i64;
    let day_micros = days * 86_400 * 1_000_000;

    let Some(tp) = time_part else {
        return Some(day_micros);
    };

    // Parse HH:MM:SS[.ffffff]
    let (hms, frac) = if let Some(dot) = tp.find('.') {
        (&tp[..dot], Some(&tp[dot+1..]))
    } else {
        (tp, None)
    };
    let parts: Vec<&str> = hms.splitn(3, ':').collect();
    if parts.len() < 2 { return None; }
    let h: i64 = parts[0].parse().ok()?;
    let m: i64 = parts[1].parse().ok()?;
    let sec: i64 = if parts.len() == 3 { parts[2].parse().ok()? } else { 0 };

    let time_micros = h * 3600 * 1_000_000 + m * 60 * 1_000_000 + sec * 1_000_000;
    let frac_micros: i64 = if let Some(f) = frac {
        // Pad or truncate to 6 digits
        let padded = format!("{:0<6}", &f[..f.len().min(6)]);
        padded.parse().ok()?
    } else {
        0
    };

    Some(day_micros + time_micros + frac_micros)
}

/// Parse a decimal string `"12.34"` given a scale into an i128.
pub fn parse_decimal(s: &str, scale: i8) -> Option<i128> {
    let s = s.trim();
    let negative = s.starts_with('-');
    let s = if negative { &s[1..] } else { s };

    let (int_str, frac_str) = if let Some(dot) = s.find('.') {
        (&s[..dot], Some(&s[dot+1..]))
    } else {
        (s, None)
    };

    let int_part: i128 = int_str.parse().ok()?;
    let factor = 10i128.pow(scale as u32);

    let frac_part: i128 = if let Some(f) = frac_str {
        let padded = format!("{:0<width$}", &f[..f.len().min(scale as usize)], width = scale as usize);
        padded.parse().ok()?
    } else {
        0
    };

    let v = int_part * factor + frac_part;
    Some(if negative { -v } else { v })
}

#[derive(Debug, Error)]
pub enum FalconError {
    #[error("column '{0}' not found")]
    ColumnNotFound(String),

    #[error("type mismatch: expected {expected}, got {got}")]
    TypeMismatch { expected: DataType, got: DataType },

    #[error("index out of bounds: index {index}, len {len}")]
    IndexOutOfBounds { index: usize, len: usize },

    #[error("parse error: {0}")]
    ParseError(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid query: {0}")]
    InvalidQuery(String),

    #[error("division by zero")]
    DivisionByZero,

    #[error("overflow in arithmetic operation")]
    Overflow,

    #[error("constraint violation: {0}")]
    ConstraintViolation(String),
}

pub type FalconResult<T> = Result<T, FalconError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_roundtrip() {
        assert_eq!(ymd_to_days(1970, 1, 1), 0);
        assert_eq!(days_to_ymd(0), (1970, 1, 1));
    }

    #[test]
    fn date_roundtrip() {
        for (y, m, d) in [(2024, 1, 15), (2000, 2, 29), (1999, 12, 31), (1970, 1, 1)] {
            let days = ymd_to_days(y, m, d);
            assert_eq!(days_to_ymd(days), (y, m, d), "roundtrip failed for {y}-{m:02}-{d:02}");
        }
    }

    #[test]
    fn parse_date_basic() {
        let d = parse_date("2024-01-15").unwrap();
        let (y, m, day) = days_to_ymd(d);
        assert_eq!((y, m, day), (2024, 1, 15));
    }

    #[test]
    fn parse_timestamp_basic() {
        let ts = parse_timestamp("2024-01-15 10:30:00").unwrap();
        let d = parse_timestamp("2024-01-15").unwrap();
        assert_eq!(ts - d, (10 * 3600 + 30 * 60) * 1_000_000);
    }
}
