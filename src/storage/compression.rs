use lz4_flex::{compress_prepend_size, decompress_size_prepended};

use super::{
    column::{BooleanColumn, Column, Decimal128Column, PrimitiveColumn, Utf8Column},
    types::{DataType, FalconError, FalconResult},
};

// ─── Compressed column ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CompressedColumn {
    pub data_type: DataType,
    pub encoding:  Encoding,
    pub null_mask: Vec<u8>,
    pub null_count: usize,
    pub row_count:  usize,
    pub data:       Vec<u8>,
    pub dict:       Option<Vec<u8>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Encoding {
    /// Raw bytes, LZ4-compressed.
    Lz4Raw,
    /// Utf8 dictionary: values replaced by u32 indices, dict LZ4-compressed.
    DictionaryLz4,
    /// Boolean packed into bits, LZ4-compressed.
    BitPackedLz4,
    /// Delta-of-delta encoding for monotone/sequential integers, then LZ4.
    DeltaLz4,
    /// Run-length encoding for low-cardinality integers, then LZ4.
    RleLz4,
}

// ─── Compress ─────────────────────────────────────────────────────────────────

pub fn compress_column(col: &Column) -> FalconResult<CompressedColumn> {
    match col {
        Column::Boolean(c)    => compress_boolean(c),
        Column::Int32(c)      => compress_int32_smart(c),
        Column::Int64(c)      => compress_int64_smart(c),
        Column::Float32(c)    => compress_primitive(c, DataType::Float32),
        Column::Float64(c)    => compress_primitive(c, DataType::Float64),
        Column::Utf8(c)       => compress_utf8(c),
        Column::Date32(c)     => compress_int32_smart_dt(c, DataType::Date32),
        Column::Timestamp(c)  => compress_int64_smart_dt(c, DataType::Timestamp),
        Column::Decimal128(c) => compress_primitive(
            &c.inner,
            DataType::Decimal128(c.precision, c.scale),
        ),
    }
}

fn compress_int32_smart(col: &PrimitiveColumn<i32>) -> FalconResult<CompressedColumn> {
    compress_int32_smart_dt(col, DataType::Int32)
}

fn compress_int32_smart_dt(col: &PrimitiveColumn<i32>, dt: DataType) -> FalconResult<CompressedColumn> {
    let vals: Vec<Option<i32>> = col.iter().collect();
    let non_null: Vec<i32> = vals.iter().filter_map(|v| *v).collect();
    if non_null.len() >= 8 && is_rle_better_i32(&non_null) {
        compress_rle_i32(col, dt)
    } else if non_null.len() >= 8 && is_delta_better_i32(&non_null) {
        compress_delta_i32(col, dt)
    } else {
        compress_primitive(col, dt)
    }
}

fn compress_int64_smart(col: &PrimitiveColumn<i64>) -> FalconResult<CompressedColumn> {
    compress_int64_smart_dt(col, DataType::Int64)
}

fn compress_int64_smart_dt(col: &PrimitiveColumn<i64>, dt: DataType) -> FalconResult<CompressedColumn> {
    let vals: Vec<Option<i64>> = col.iter().collect();
    let non_null: Vec<i64> = vals.iter().filter_map(|v| *v).collect();
    if non_null.len() >= 8 && is_rle_better_i64(&non_null) {
        compress_rle_i64(col, dt)
    } else if non_null.len() >= 8 && is_delta_better_i64(&non_null) {
        compress_delta_i64(col, dt)
    } else {
        compress_primitive(col, dt)
    }
}

// Returns true if RLE would compress better (many repeated runs).
fn is_rle_better_i32(vals: &[i32]) -> bool {
    let runs = count_runs_i32(vals);
    runs < vals.len() / 4  // at least 4:1 run compression
}

fn is_rle_better_i64(vals: &[i64]) -> bool {
    let runs = count_runs_i64(vals);
    runs < vals.len() / 4
}

fn count_runs_i32(vals: &[i32]) -> usize {
    if vals.is_empty() { return 0; }
    let mut runs = 1usize;
    for i in 1..vals.len() { if vals[i] != vals[i-1] { runs += 1; } }
    runs
}

fn count_runs_i64(vals: &[i64]) -> usize {
    if vals.is_empty() { return 0; }
    let mut runs = 1usize;
    for i in 1..vals.len() { if vals[i] != vals[i-1] { runs += 1; } }
    runs
}

// Returns true if delta encoding reduces bit-width significantly.
fn is_delta_better_i32(vals: &[i32]) -> bool {
    if vals.len() < 2 { return false; }
    let deltas: Vec<i64> = vals.windows(2).map(|w| (w[1] as i64) - (w[0] as i64)).collect();
    let max_delta = deltas.iter().map(|d| d.unsigned_abs()).max().unwrap_or(0);
    let max_val   = vals.iter().map(|v| v.unsigned_abs() as u64).max().unwrap_or(0);
    max_delta < max_val / 4
}

fn is_delta_better_i64(vals: &[i64]) -> bool {
    if vals.len() < 2 { return false; }
    let deltas: Vec<i64> = vals.windows(2).map(|w| w[1].wrapping_sub(w[0])).collect();
    let max_delta = deltas.iter().map(|d| d.unsigned_abs()).max().unwrap_or(0);
    let max_val   = vals.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
    max_delta < max_val / 4
}

// ─── Delta encoding ───────────────────────────────────────────────────────────

fn compress_delta_i32(col: &PrimitiveColumn<i32>, dt: DataType) -> FalconResult<CompressedColumn> {
    let mut nulls      = Vec::with_capacity(col.len());
    let mut raw_vals   = Vec::with_capacity(col.len());
    let mut null_count = 0usize;

    for v in col.iter() {
        match v {
            Some(val) => { nulls.push(false); raw_vals.push(val); }
            None      => { nulls.push(true);  raw_vals.push(0i32); null_count += 1; }
        }
    }

    // Store first value + deltas as i64 (prevents overflow on subtraction).
    let mut encoded: Vec<i64> = Vec::with_capacity(raw_vals.len());
    if !raw_vals.is_empty() {
        encoded.push(raw_vals[0] as i64);
        for i in 1..raw_vals.len() {
            encoded.push((raw_vals[i] as i64) - (raw_vals[i-1] as i64));
        }
    }

    let bytes      = bytemuck::cast_slice::<i64, u8>(&encoded);
    let compressed = compress_prepend_size(bytes);

    Ok(CompressedColumn {
        data_type:  dt,
        encoding:   Encoding::DeltaLz4,
        null_mask:  null_mask_encode(&nulls),
        null_count,
        row_count:  col.len(),
        data:       compressed,
        dict:       None,
    })
}

fn compress_delta_i64(col: &PrimitiveColumn<i64>, dt: DataType) -> FalconResult<CompressedColumn> {
    let mut nulls      = Vec::with_capacity(col.len());
    let mut raw_vals   = Vec::with_capacity(col.len());
    let mut null_count = 0usize;

    for v in col.iter() {
        match v {
            Some(val) => { nulls.push(false); raw_vals.push(val); }
            None      => { nulls.push(true);  raw_vals.push(0i64); null_count += 1; }
        }
    }

    let mut encoded: Vec<i64> = Vec::with_capacity(raw_vals.len());
    if !raw_vals.is_empty() {
        encoded.push(raw_vals[0]);
        for i in 1..raw_vals.len() {
            encoded.push(raw_vals[i].wrapping_sub(raw_vals[i-1]));
        }
    }

    let bytes      = bytemuck::cast_slice::<i64, u8>(&encoded);
    let compressed = compress_prepend_size(bytes);

    Ok(CompressedColumn {
        data_type:  dt,
        encoding:   Encoding::DeltaLz4,
        null_mask:  null_mask_encode(&nulls),
        null_count,
        row_count:  col.len(),
        data:       compressed,
        dict:       None,
    })
}

// ─── RLE encoding ─────────────────────────────────────────────────────────────
// Format: [i64 value][u32 run_length] pairs, LZ4-compressed.

fn compress_rle_i32(col: &PrimitiveColumn<i32>, dt: DataType) -> FalconResult<CompressedColumn> {
    let mut nulls      = Vec::with_capacity(col.len());
    let mut raw_vals   = Vec::with_capacity(col.len());
    let mut null_count = 0usize;

    for v in col.iter() {
        match v {
            Some(val) => { nulls.push(false); raw_vals.push(val as i64); }
            None      => { nulls.push(true);  raw_vals.push(i64::MIN); null_count += 1; }
        }
    }

    let rle = encode_rle_i64(&raw_vals);
    let bytes = bytemuck::cast_slice::<i64, u8>(&rle);
    let compressed = compress_prepend_size(bytes);

    Ok(CompressedColumn {
        data_type:  dt,
        encoding:   Encoding::RleLz4,
        null_mask:  null_mask_encode(&nulls),
        null_count,
        row_count:  col.len(),
        data:       compressed,
        dict:       None,
    })
}

fn compress_rle_i64(col: &PrimitiveColumn<i64>, dt: DataType) -> FalconResult<CompressedColumn> {
    let mut nulls      = Vec::with_capacity(col.len());
    let mut raw_vals   = Vec::with_capacity(col.len());
    let mut null_count = 0usize;

    for v in col.iter() {
        match v {
            Some(val) => { nulls.push(false); raw_vals.push(val); }
            None      => { nulls.push(true);  raw_vals.push(i64::MIN); null_count += 1; }
        }
    }

    let rle = encode_rle_i64(&raw_vals);
    let bytes = bytemuck::cast_slice::<i64, u8>(&rle);
    let compressed = compress_prepend_size(bytes);

    Ok(CompressedColumn {
        data_type:  dt,
        encoding:   Encoding::RleLz4,
        null_mask:  null_mask_encode(&nulls),
        null_count,
        row_count:  col.len(),
        data:       compressed,
        dict:       None,
    })
}

/// Encode as interleaved [value, run_length] i64 pairs.
fn encode_rle_i64(vals: &[i64]) -> Vec<i64> {
    let mut out = Vec::new();
    if vals.is_empty() { return out; }
    let mut cur = vals[0];
    let mut run = 1i64;
    for &v in &vals[1..] {
        if v == cur {
            run += 1;
        } else {
            out.push(cur);
            out.push(run);
            cur = v;
            run = 1;
        }
    }
    out.push(cur);
    out.push(run);
    out
}

fn decode_rle_i64(pairs: &[i64]) -> Vec<i64> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 1 < pairs.len() {
        let val = pairs[i];
        let run = pairs[i+1].max(0) as usize;
        for _ in 0..run { out.push(val); }
        i += 2;
    }
    out
}

fn null_mask_encode(nulls: &[bool]) -> Vec<u8> {
    // Pack booleans into bytes, 8 per byte.
    let byte_count = nulls.len().div_ceil(8);
    let mut mask = vec![0u8; byte_count];
    for (i, &is_null) in nulls.iter().enumerate() {
        if is_null {
            mask[i / 8] |= 1 << (i % 8);
        }
    }
    mask
}

fn compress_primitive<T: Copy>(
    col: &PrimitiveColumn<T>,
    dt:  DataType,
) -> FalconResult<CompressedColumn>
where
    T: bytemuck::Pod + Default,
{
    let mut nulls     = Vec::with_capacity(col.len());
    let mut raw_vals: Vec<T> = Vec::with_capacity(col.len());
    let mut null_count = 0usize;

    for v in col.iter() {
        match v {
            Some(val) => { nulls.push(false); raw_vals.push(val); }
            None      => { nulls.push(true);  raw_vals.push(bytemuck::Zeroable::zeroed()); null_count += 1; }
        }
    }

    let bytes     = bytemuck::cast_slice::<T, u8>(&raw_vals);
    let compressed = compress_prepend_size(bytes);

    Ok(CompressedColumn {
        data_type:  dt,
        encoding:   Encoding::Lz4Raw,
        null_mask:  null_mask_encode(&nulls),
        null_count,
        row_count:  col.len(),
        data:       compressed,
        dict:       None,
    })
}

fn compress_boolean(col: &BooleanColumn) -> FalconResult<CompressedColumn> {
    let mut nulls      = Vec::with_capacity(col.len());
    let mut bits       = Vec::with_capacity(col.len());
    let mut null_count = 0usize;

    for v in col.values.iter() {
        match v {
            Some(b) => { nulls.push(false); bits.push(*b); }
            None    => { nulls.push(true);  bits.push(false); null_count += 1; }
        }
    }

    let packed    = pack_bits(&bits);
    let compressed = compress_prepend_size(&packed);

    Ok(CompressedColumn {
        data_type:  DataType::Boolean,
        encoding:   Encoding::BitPackedLz4,
        null_mask:  null_mask_encode(&nulls),
        null_count,
        row_count:  col.len(),
        data:       compressed,
        dict:       None,
    })
}

fn compress_utf8(col: &Utf8Column) -> FalconResult<CompressedColumn> {
    let mut nulls      = Vec::with_capacity(col.len());
    let mut null_count = 0usize;

    // Build dictionary: unique strings → index.
    let mut dict_map: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
    let mut dict_strs: Vec<&str> = Vec::new();

    for v in col.iter() {
        match v {
            Some(s) => {
                nulls.push(false);
                if !dict_map.contains_key(s) {
                    dict_map.insert(s, dict_strs.len() as u32);
                    dict_strs.push(s);
                }
            }
            None => { nulls.push(true); null_count += 1; }
        }
    }

    // Encode indices.
    let mut indices = Vec::with_capacity(col.len());
    for v in col.iter() {
        match v {
            Some(s) => indices.push(*dict_map.get(s).unwrap()),
            None    => indices.push(u32::MAX),
        }
    }

    let idx_bytes  = bytemuck::cast_slice::<u32, u8>(&indices);
    let compressed = compress_prepend_size(idx_bytes);

    // Serialize dictionary as: [u32 count][u32 len, bytes...] per entry.
    let mut dict_bytes: Vec<u8> = Vec::new();
    dict_bytes.extend_from_slice(&(dict_strs.len() as u32).to_le_bytes());
    for s in &dict_strs {
        let b = s.as_bytes();
        dict_bytes.extend_from_slice(&(b.len() as u32).to_le_bytes());
        dict_bytes.extend_from_slice(b);
    }
    let dict_compressed = compress_prepend_size(&dict_bytes);

    Ok(CompressedColumn {
        data_type:  DataType::Utf8,
        encoding:   Encoding::DictionaryLz4,
        null_mask:  null_mask_encode(&nulls),
        null_count,
        row_count:  col.len(),
        data:       compressed,
        dict:       Some(dict_compressed),
    })
}

// ─── Decompress ───────────────────────────────────────────────────────────────

pub fn decompress_column(comp: &CompressedColumn) -> FalconResult<Column> {
    let nulls = null_mask_decode(&comp.null_mask, comp.row_count);

    match comp.encoding {
        Encoding::Lz4Raw        => decompress_primitive(comp, &nulls),
        Encoding::BitPackedLz4  => decompress_boolean(comp, &nulls),
        Encoding::DictionaryLz4 => decompress_utf8(comp, &nulls),
        Encoding::DeltaLz4      => decompress_delta(comp, &nulls),
        Encoding::RleLz4        => decompress_rle(comp, &nulls),
    }
}

fn decompress_delta(comp: &CompressedColumn, nulls: &[bool]) -> FalconResult<Column> {
    let raw = decompress_size_prepended(&comp.data)
        .map_err(|e| FalconError::ParseError(format!("lz4 decompress failed: {e}")))?;
    let deltas: &[i64] = bytemuck::cast_slice(&raw);

    // Reconstruct original values from first value + cumulative deltas.
    let mut vals: Vec<i64> = Vec::with_capacity(deltas.len());
    if !deltas.is_empty() {
        vals.push(deltas[0]);
        for i in 1..deltas.len() {
            vals.push(vals[i-1].wrapping_add(deltas[i]));
        }
    }

    Ok(match &comp.data_type {
        DataType::Int32 => Column::Int32(PrimitiveColumn::new(
            vals.iter().zip(nulls).map(|(&v, &n)| if n { None } else { Some(v as i32) }).collect(),
        )),
        DataType::Int64 => Column::Int64(PrimitiveColumn::new(
            vals.iter().zip(nulls).map(|(&v, &n)| if n { None } else { Some(v) }).collect(),
        )),
        DataType::Date32 => Column::Date32(PrimitiveColumn::new(
            vals.iter().zip(nulls).map(|(&v, &n)| if n { None } else { Some(v as i32) }).collect(),
        )),
        DataType::Timestamp => Column::Timestamp(PrimitiveColumn::new(
            vals.iter().zip(nulls).map(|(&v, &n)| if n { None } else { Some(v) }).collect(),
        )),
        _ => return Err(FalconError::ParseError("unexpected type in delta decompression".into())),
    })
}

fn decompress_rle(comp: &CompressedColumn, nulls: &[bool]) -> FalconResult<Column> {
    let raw = decompress_size_prepended(&comp.data)
        .map_err(|e| FalconError::ParseError(format!("lz4 decompress failed: {e}")))?;
    let pairs: &[i64] = bytemuck::cast_slice(&raw);
    let vals = decode_rle_i64(pairs);

    Ok(match &comp.data_type {
        DataType::Int32 => Column::Int32(PrimitiveColumn::new(
            vals.iter().zip(nulls).map(|(&v, &n)| if n { None } else { Some(v as i32) }).collect(),
        )),
        DataType::Int64 => Column::Int64(PrimitiveColumn::new(
            vals.iter().zip(nulls).map(|(&v, &n)| if n { None } else { Some(v) }).collect(),
        )),
        DataType::Date32 => Column::Date32(PrimitiveColumn::new(
            vals.iter().zip(nulls).map(|(&v, &n)| if n { None } else { Some(v as i32) }).collect(),
        )),
        DataType::Timestamp => Column::Timestamp(PrimitiveColumn::new(
            vals.iter().zip(nulls).map(|(&v, &n)| if n { None } else { Some(v) }).collect(),
        )),
        _ => return Err(FalconError::ParseError("unexpected type in RLE decompression".into())),
    })
}

fn null_mask_decode(mask: &[u8], row_count: usize) -> Vec<bool> {
    let mut nulls = Vec::with_capacity(row_count);
    for i in 0..row_count {
        nulls.push((mask[i / 8] >> (i % 8)) & 1 == 1);
    }
    nulls
}

fn decompress_primitive(comp: &CompressedColumn, nulls: &[bool]) -> FalconResult<Column> {
    let raw = decompress_size_prepended(&comp.data)
        .map_err(|e| FalconError::ParseError(format!("lz4 decompress failed: {e}")))?;

    Ok(match comp.data_type {
        DataType::Int32 => {
            let vals: &[i32] = bytemuck::cast_slice(&raw);
            Column::Int32(PrimitiveColumn::new(
                vals.iter().zip(nulls).map(|(&v, &n)| if n { None } else { Some(v) }).collect(),
            ))
        }
        DataType::Int64 => {
            let vals: &[i64] = bytemuck::cast_slice(&raw);
            Column::Int64(PrimitiveColumn::new(
                vals.iter().zip(nulls).map(|(&v, &n)| if n { None } else { Some(v) }).collect(),
            ))
        }
        DataType::Float32 => {
            let vals: &[f32] = bytemuck::cast_slice(&raw);
            Column::Float32(PrimitiveColumn::new(
                vals.iter().zip(nulls).map(|(&v, &n)| if n { None } else { Some(v) }).collect(),
            ))
        }
        DataType::Float64 => {
            let vals: &[f64] = bytemuck::cast_slice(&raw);
            Column::Float64(PrimitiveColumn::new(
                vals.iter().zip(nulls).map(|(&v, &n)| if n { None } else { Some(v) }).collect(),
            ))
        }
        DataType::Date32 => {
            let vals: &[i32] = bytemuck::cast_slice(&raw);
            Column::Date32(PrimitiveColumn::new(
                vals.iter().zip(nulls).map(|(&v, &n)| if n { None } else { Some(v) }).collect(),
            ))
        }
        DataType::Timestamp => {
            let vals: &[i64] = bytemuck::cast_slice(&raw);
            Column::Timestamp(PrimitiveColumn::new(
                vals.iter().zip(nulls).map(|(&v, &n)| if n { None } else { Some(v) }).collect(),
            ))
        }
        DataType::Decimal128(p, s) => {
            // Read 16 bytes per value in LE order (avoids alignment requirements).
            let vals: Vec<i128> = raw.chunks_exact(16)
                .map(|chunk| i128::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            Column::Decimal128(Decimal128Column::new(
                vals.iter().zip(nulls).map(|(&v, &n)| if n { None } else { Some(v) }).collect(),
                p, s,
            ))
        }
        _ => return Err(FalconError::ParseError("unexpected type in primitive decompression".into())),
    })
}

fn decompress_boolean(comp: &CompressedColumn, nulls: &[bool]) -> FalconResult<Column> {
    let raw = decompress_size_prepended(&comp.data)
        .map_err(|e| FalconError::ParseError(format!("lz4 decompress failed: {e}")))?;
    let bits = unpack_bits(&raw, comp.row_count);
    Ok(Column::Boolean(BooleanColumn::new(
        bits.iter().zip(nulls).map(|(&b, &n)| if n { None } else { Some(b) }).collect(),
    )))
}

fn decompress_utf8(comp: &CompressedColumn, nulls: &[bool]) -> FalconResult<Column> {
    let dict_bytes = decompress_size_prepended(comp.dict.as_deref().unwrap_or(&[]))
        .map_err(|e| FalconError::ParseError(format!("lz4 dict decompress failed: {e}")))?;

    let dict = decode_dict(&dict_bytes)?;

    let raw = decompress_size_prepended(&comp.data)
        .map_err(|e| FalconError::ParseError(format!("lz4 decompress failed: {e}")))?;

    let indices: &[u32] = bytemuck::cast_slice(&raw);
    let vals: Vec<Option<String>> = indices.iter().zip(nulls).map(|(&idx, &is_null)| {
        if is_null { None }
        else { dict.get(idx as usize).cloned() }
    }).collect();

    Ok(Column::Utf8(Utf8Column::new(vals)))
}

fn decode_dict(bytes: &[u8]) -> FalconResult<Vec<String>> {
    if bytes.len() < 4 {
        return Ok(Vec::new());
    }
    let count = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let mut pos = 4usize;
    let mut dict = Vec::with_capacity(count);

    for _ in 0..count {
        let len = u32::from_le_bytes(
            bytes[pos..pos+4].try_into()
                .map_err(|_| FalconError::ParseError("dict decode: truncated length".into()))?,
        ) as usize;
        pos += 4;
        let s = std::str::from_utf8(&bytes[pos..pos+len])
            .map_err(|_| FalconError::ParseError("dict decode: invalid utf8".into()))?;
        dict.push(s.to_string());
        pos += len;
    }

    Ok(dict)
}

// ─── Bit packing ─────────────────────────────────────────────────────────────

fn pack_bits(bits: &[bool]) -> Vec<u8> {
    let byte_count = bits.len().div_ceil(8);
    let mut bytes  = vec![0u8; byte_count];
    for (i, &b) in bits.iter().enumerate() {
        if b { bytes[i / 8] |= 1 << (i % 8); }
    }
    bytes
}

fn unpack_bits(bytes: &[u8], count: usize) -> Vec<bool> {
    (0..count).map(|i| (bytes[i / 8] >> (i % 8)) & 1 == 1).collect()
}

// ─── Compression stats ────────────────────────────────────────────────────────

pub struct CompressionStats {
    pub original_bytes:   usize,
    pub compressed_bytes: usize,
    pub ratio:            f64,
    pub encoding:         Encoding,
}

impl CompressionStats {
    pub fn from(original: &Column, compressed: &CompressedColumn) -> Self {
        let original_bytes = estimate_raw_size(original);
        let compressed_bytes = compressed.data.len()
            + compressed.null_mask.len()
            + compressed.dict.as_deref().map(|d| d.len()).unwrap_or(0);
        let ratio = compressed_bytes as f64 / original_bytes.max(1) as f64;
        Self { original_bytes, compressed_bytes, ratio, encoding: compressed.encoding.clone() }
    }
}

fn estimate_raw_size(col: &Column) -> usize {
    match col {
        Column::Boolean(c)    => c.values.len(),
        Column::Int32(c)      => c.len() * 4,
        Column::Int64(c)      => c.len() * 8,
        Column::Float32(c)    => c.len() * 4,
        Column::Float64(c)    => c.len() * 8,
        Column::Date32(c)     => c.len() * 4,
        Column::Timestamp(c)  => c.len() * 8,
        Column::Decimal128(c) => c.len() * 16,
        Column::Utf8(c)       => c.iter()
            .map(|v| v.as_deref().map(|s| s.len()).unwrap_or(0) + 8)
            .sum(),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::column::ColumnBuilder;

    #[test]
    fn roundtrip_int64() {
        let col = ColumnBuilder::int64(vec![Some(1), None, Some(3), Some(-99), None]);
        let comp = compress_column(&col).unwrap();
        let back = decompress_column(&comp).unwrap();
        for i in 0..5 {
            assert_eq!(col.get(i).unwrap(), back.get(i).unwrap());
        }
    }

    #[test]
    fn roundtrip_float64() {
        let col = ColumnBuilder::float64(vec![Some(3.14), None, Some(-2.71)]);
        let comp = compress_column(&col).unwrap();
        let back = decompress_column(&comp).unwrap();
        for i in 0..3 {
            assert_eq!(col.get(i).unwrap(), back.get(i).unwrap());
        }
    }

    #[test]
    fn roundtrip_utf8_dictionary() {
        let col = ColumnBuilder::utf8_from_str(vec![
            Some("eng"), Some("hr"), Some("eng"), None, Some("hr"),
        ]);
        let comp = compress_column(&col).unwrap();
        assert_eq!(comp.encoding, Encoding::DictionaryLz4);
        let back = decompress_column(&comp).unwrap();
        for i in 0..5 {
            assert_eq!(col.get(i).unwrap(), back.get(i).unwrap());
        }
    }

    #[test]
    fn roundtrip_boolean() {
        let col = ColumnBuilder::boolean(vec![Some(true), None, Some(false), Some(true)]);
        let comp = compress_column(&col).unwrap();
        assert_eq!(comp.encoding, Encoding::BitPackedLz4);
        let back = decompress_column(&comp).unwrap();
        for i in 0..4 {
            assert_eq!(col.get(i).unwrap(), back.get(i).unwrap());
        }
    }

    #[test]
    fn null_count_preserved() {
        let col  = ColumnBuilder::int64(vec![Some(1), None, None, Some(4)]);
        let comp = compress_column(&col).unwrap();
        assert_eq!(comp.null_count, 2);
    }

    #[test]
    fn compression_reduces_size_for_repetitive_data() {
        let col = ColumnBuilder::utf8_from_str(
            (0..1000).map(|i| Some(["eng", "hr", "finance"][i % 3])).collect(),
        );
        let comp  = compress_column(&col).unwrap();
        let stats = CompressionStats::from(&col, &comp);
        assert!(stats.ratio < 0.5, "expected >50% compression, got {:.2}", stats.ratio);
    }
}
