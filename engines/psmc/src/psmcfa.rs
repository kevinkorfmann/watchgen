use crate::error::PsmcError;
use std::io::{BufRead, Write};

/// A single PSMCFA record (one sequence).
#[derive(Debug, Clone)]
pub struct PsmcfaRecord {
    pub name: String,
    /// 0 = homozygous (K), 1 = heterozygous (T), 2 = missing (N)
    pub seq: Vec<u8>,
}

/// Read PSMCFA format from a reader.
///
/// Format: FASTA-like with header lines starting with '>'.
/// Characters: T = heterozygous (1), K = homozygous (0), N = missing (2).
pub fn read_psmcfa<R: BufRead>(reader: R) -> Result<Vec<PsmcfaRecord>, PsmcError> {
    let mut records = Vec::new();
    let mut current_name: Option<String> = None;
    let mut current_seq: Vec<u8> = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim_end();
        if line.is_empty() {
            continue;
        }
        if let Some(name) = line.strip_prefix('>') {
            if let Some(prev_name) = current_name.take() {
                if current_seq.is_empty() {
                    return Err(PsmcError::InvalidPsmcfa(format!(
                        "empty sequence for record '{prev_name}'"
                    )));
                }
                records.push(PsmcfaRecord {
                    name: prev_name,
                    seq: std::mem::take(&mut current_seq),
                });
            }
            current_name = Some(name.to_string());
        } else {
            for ch in line.chars() {
                let val = match ch {
                    'T' => 1,
                    'K' => 0,
                    'N' => 2,
                    _ => {
                        return Err(PsmcError::InvalidPsmcfa(format!(
                            "unexpected character: '{ch}'"
                        )));
                    }
                };
                current_seq.push(val);
            }
        }
    }

    if let Some(name) = current_name {
        if current_seq.is_empty() {
            return Err(PsmcError::InvalidPsmcfa(format!(
                "empty sequence for record '{name}'"
            )));
        }
        records.push(PsmcfaRecord {
            name,
            seq: current_seq,
        });
    }

    Ok(records)
}

/// Write PSMCFA format to a writer (80 chars per line).
pub fn write_psmcfa<W: Write>(
    writer: &mut W,
    records: &[PsmcfaRecord],
) -> Result<(), PsmcError> {
    for record in records {
        writeln!(writer, ">{}", record.name)?;
        let chars: Vec<char> = record
            .seq
            .iter()
            .map(|&v| match v {
                0 => 'K',
                1 => 'T',
                _ => 'N',
            })
            .collect();
        for chunk in chars.chunks(80) {
            let line: String = chunk.iter().collect();
            writeln!(writer, "{line}")?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_read_simple() {
        let data = b">seq1\nTTKKNT\n>seq2\nKKK\n";
        let records = read_psmcfa(Cursor::new(data)).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].name, "seq1");
        assert_eq!(records[0].seq, vec![1, 1, 0, 0, 2, 1]);
        assert_eq!(records[1].name, "seq2");
        assert_eq!(records[1].seq, vec![0, 0, 0]);
    }

    #[test]
    fn test_read_multiline() {
        let data = b">seq1\nTTKK\nNTKK\n";
        let records = read_psmcfa(Cursor::new(data)).unwrap();
        assert_eq!(records[0].seq, vec![1, 1, 0, 0, 2, 1, 0, 0]);
    }

    #[test]
    fn test_invalid_char() {
        let data = b">seq1\nTTXK\n";
        assert!(read_psmcfa(Cursor::new(data)).is_err());
    }

    #[test]
    fn test_roundtrip() {
        let original = vec![PsmcfaRecord {
            name: "test".into(),
            seq: vec![0, 1, 1, 0, 2, 1],
        }];
        let mut buf = Vec::new();
        write_psmcfa(&mut buf, &original).unwrap();
        let recovered = read_psmcfa(Cursor::new(&buf)).unwrap();
        assert_eq!(recovered[0].seq, original[0].seq);
    }
}
