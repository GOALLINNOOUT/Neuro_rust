use std::io::{self, Read, Write};

pub(crate) fn write_u32(writer: &mut dyn Write, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

pub(crate) fn read_u32(reader: &mut dyn Read) -> io::Result<u32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

pub(crate) fn write_usize(writer: &mut dyn Write, value: usize) -> io::Result<()> {
    let value = u64::try_from(value)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "value does not fit u64"))?;
    writer.write_all(&value.to_le_bytes())
}

pub(crate) fn read_usize(reader: &mut dyn Read) -> io::Result<usize> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    usize::try_from(u64::from_le_bytes(bytes))
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "value does not fit usize"))
}

pub(crate) fn write_f64(writer: &mut dyn Write, value: f64) -> io::Result<()> {
    if !value.is_finite() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "checkpoint cannot store a non-finite weight",
        ));
    }
    writer.write_all(&value.to_le_bytes())
}

pub(crate) fn read_f64(reader: &mut dyn Read) -> io::Result<f64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    let value = f64::from_le_bytes(bytes);
    if !value.is_finite() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "checkpoint contains a non-finite weight",
        ));
    }
    Ok(value)
}

pub(crate) fn write_f64s(writer: &mut dyn Write, values: &[f64]) -> io::Result<()> {
    for &value in values {
        write_f64(writer, value)?;
    }
    Ok(())
}

pub(crate) fn read_f64s(reader: &mut dyn Read, count: usize) -> io::Result<Vec<f64>> {
    (0..count).map(|_| read_f64(reader)).collect()
}

pub(crate) fn write_matrix(writer: &mut dyn Write, matrix: &[Vec<f64>]) -> io::Result<()> {
    for row in matrix {
        write_f64s(writer, row)?;
    }
    Ok(())
}

pub(crate) fn read_matrix(
    reader: &mut dyn Read,
    rows: usize,
    columns: usize,
) -> io::Result<Vec<Vec<f64>>> {
    let count = rows
        .checked_mul(columns)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "matrix size overflow"))?;
    let values = read_f64s(reader, count)?;
    Ok(values
        .chunks(columns.max(1))
        .map(|row| row.to_vec())
        .collect())
}

pub(crate) fn write_string(writer: &mut dyn Write, value: &str) -> io::Result<()> {
    write_usize(writer, value.len())?;
    writer.write_all(value.as_bytes())
}

pub(crate) fn read_string(reader: &mut dyn Read, max_bytes: usize) -> io::Result<String> {
    let length = read_usize(reader)?;
    if length > max_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "checkpoint string length exceeds file size",
        ));
    }
    let mut bytes = vec![0; length];
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "checkpoint token is not UTF-8"))
}
