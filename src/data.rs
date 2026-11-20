use std::fs;

/// Loads a numeric column. Header detection follows the requested rule: if every
/// field of the first row parses as a number, it is data; otherwise it is a header.
pub fn load_column(path: &str, selector: &str) -> Result<Vec<f64>, String> {
    let content = fs::read_to_string(path).map_err(|e| format!("could not read '{path}': {e}"))?;
    let mut rows = content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(parse_row);
    let first = rows.next().ok_or_else(|| format!("'{path}' is empty"))??;
    let has_header = !first.iter().all(|cell| cell.trim().parse::<f64>().is_ok());
    let column = if let Ok(index) = selector.parse::<usize>() {
        index
    } else if has_header {
        first
            .iter()
            .position(|name| name.trim() == selector)
            .ok_or_else(|| format!("column '{selector}' was not found in the header"))?
    } else {
        return Err(format!(
            "column '{selector}' is not a 0-based index and the file has no header"
        ));
    };
    let data_rows: Box<dyn Iterator<Item = Result<Vec<String>, String>>> = if has_header {
        Box::new(rows)
    } else {
        Box::new(std::iter::once(Ok(first)).chain(rows))
    };
    let mut values = Vec::new();
    for (row_number, row) in data_rows.enumerate() {
        let row = row?;
        match row.get(column).map(|s| s.trim()).filter(|s| !s.is_empty()) {
            Some(cell) => match cell.parse::<f64>() {
                Ok(value) if value.is_finite() => values.push(value),
                _ => eprintln!(
                    "Warning: skipping non-numeric value in column {column} at data row {}",
                    row_number + 1
                ),
            },
            None => eprintln!(
                "Warning: skipping missing or empty value in column {column} at data row {}",
                row_number + 1
            ),
        }
    }
    if values.is_empty() {
        return Err(format!(
            "column {column} contains zero valid numeric values"
        ));
    }
    Ok(values)
}

/// Parses one CSV record, including quoted fields and doubled quote escapes.
fn parse_row(line: &str) -> Result<Vec<String>, String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut chars = line.trim_end_matches('\r').chars().peekable();
    let mut quoted = false;
    while let Some(ch) = chars.next() {
        match ch {
            '"' if quoted && chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => {
                fields.push(std::mem::take(&mut field));
            }
            _ => field.push(ch),
        }
    }
    if quoted {
        return Err("unterminated quoted field in CSV".to_string());
    }
    fields.push(field);
    Ok(fields)
}

#[cfg(test)]
mod tests {
    use super::parse_row;

    #[test]
    fn parses_quoted_csv_fields() {
        assert_eq!(
            parse_row("name,\"a,b\",\"say \"\"hi\"\"\"").unwrap(),
            ["name", "a,b", "say \"hi\""]
        );
    }
}
