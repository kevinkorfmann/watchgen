use crate::error::PsmcError;

/// Parse a PSMC pattern string into a parameter map.
///
/// Pattern format: "4+25*2+4+6"
///   - "4" means 4 atomic intervals sharing one free parameter
///   - "25*2" means 25 groups of 2 atomic intervals each (25 free params)
///
/// Returns `(par_map, n_free, n_intervals)` where:
///   - `par_map[k]` = index of the free parameter for interval k
///   - `n_free` = number of free parameters
///   - `n_intervals` = total number of atomic intervals
pub fn parse_pattern(pattern: &str) -> Result<(Vec<usize>, usize, usize), PsmcError> {
    let mut par_map = Vec::new();
    let mut free_idx = 0usize;

    for part in pattern.split('+') {
        let part = part.trim();
        if part.is_empty() {
            return Err(PsmcError::InvalidPattern(
                "empty segment in pattern".into(),
            ));
        }

        if let Some((count_str, width_str)) = part.split_once('*') {
            let count: usize = count_str
                .trim()
                .parse()
                .map_err(|_| PsmcError::InvalidPattern(format!("bad count: {count_str}")))?;
            let width: usize = width_str
                .trim()
                .parse()
                .map_err(|_| PsmcError::InvalidPattern(format!("bad width: {width_str}")))?;
            for _ in 0..count {
                for _ in 0..width {
                    par_map.push(free_idx);
                }
                free_idx += 1;
            }
        } else {
            let width: usize = part
                .parse()
                .map_err(|_| PsmcError::InvalidPattern(format!("bad width: {part}")))?;
            for _ in 0..width {
                par_map.push(free_idx);
            }
            free_idx += 1;
        }
    }

    let n_intervals = par_map.len();
    Ok((par_map, free_idx, n_intervals))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_pattern() {
        let (par_map, n_free, n_intervals) = parse_pattern("4+25*2+4+6").unwrap();
        assert_eq!(n_intervals, 64);
        assert_eq!(n_free, 28);
        assert_eq!(par_map.len(), 64);
    }

    #[test]
    fn test_all_free() {
        let (par_map, n_free, n_intervals) = parse_pattern("1+1+1+1").unwrap();
        assert_eq!(n_intervals, 4);
        assert_eq!(n_free, 4);
        assert_eq!(par_map, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_single_group() {
        let (par_map, n_free, n_intervals) = parse_pattern("10").unwrap();
        assert_eq!(n_intervals, 10);
        assert_eq!(n_free, 1);
        assert!(par_map.iter().all(|&p| p == 0));
    }

    #[test]
    fn test_repeated_groups() {
        let (_, n_free, n_intervals) = parse_pattern("5*2").unwrap();
        assert_eq!(n_intervals, 10);
        assert_eq!(n_free, 5);
    }

    #[test]
    fn test_par_map_structure() {
        let (par_map, _, _) = parse_pattern("4+25*2+4+6").unwrap();
        // First 4 intervals share parameter 0
        assert!(par_map[0..4].iter().all(|&p| p == 0));
        // Next 2 intervals share parameter 1
        assert_eq!(par_map[4], 1);
        assert_eq!(par_map[5], 1);
    }

    #[test]
    fn test_invalid_pattern() {
        assert!(parse_pattern("abc").is_err());
        assert!(parse_pattern("4+*2").is_err());
    }
}
