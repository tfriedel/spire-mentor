//! Python's `json.dumps(x, sort_keys=True)` over a [`serde_json::Value`], byte
//! for byte, plus the few dict idioms the env's Python source leans on.
//!
//! The env's merge keys are canonical JSON strings that the parity fixtures
//! record, so they must match Python's output exactly: `", "` / `": "`
//! separators, keys sorted by code point, `ensure_ascii` escapes, and
//! `float.__repr__` for floats (`1.0`, `1e+16`, `1.5e-05`). Integers and
//! floats stay distinct (`1` vs `1.0`), as do `true` and `1`.

use serde_json::{Map, Value};

/// `json.dumps(v, sort_keys=True)` (default separators, `ensure_ascii=True`).
#[must_use]
pub fn dumps(v: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, v);
    out
}

fn write_value(out: &mut String, v: &Value) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else {
                out.push_str(&py_float_repr(n.as_f64().unwrap_or(f64::NAN)));
            }
        }
        Value::String(s) => write_str(out, s),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_value(out, x);
            }
            out.push(']');
        }
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            // Python sorts str keys by code point; UTF-8 byte order is the same.
            keys.sort();
            out.push('{');
            for (i, k) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_str(out, k);
                out.push_str(": ");
                write_value(out, &m[k]);
            }
            out.push('}');
        }
    }
}

fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            ' '..='~' => out.push(ch),
            _ => {
                let mut buf = [0u16; 2];
                for unit in ch.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
}

/// Python's `repr(float)`: the shortest round-trip digits, fixed notation for
/// decimal exponents in `[-4, 16)`, else `d.ddde±XX`.
#[must_use]
pub fn py_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "NaN".to_owned();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity" } else { "-Infinity" }.to_owned();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.to_owned();
    }
    // Rust's `{:e}` prints the shortest round-trip digits as `d.ddde<exp>`.
    let sci = format!("{x:e}");
    let (mant, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(char::is_ascii_digit).collect();
    let sign = if neg { "-" } else { "" };
    if (-4..16).contains(&exp) {
        let n = i32::try_from(digits.len()).unwrap_or(i32::MAX);
        let point = exp + 1; // digits before the decimal point
        let body = if point <= 0 {
            format!("0.{}{}", "0".repeat(point.unsigned_abs() as usize), digits)
        } else if point >= n {
            format!(
                "{}{}.0",
                digits,
                "0".repeat(usize::try_from(point - n).unwrap_or(0))
            )
        } else {
            let p = usize::try_from(point).unwrap_or(0);
            format!("{}.{}", &digits[..p], &digits[p..])
        };
        format!("{sign}{body}")
    } else {
        let m = if digits.len() > 1 {
            format!("{}.{}", &digits[..1], &digits[1..])
        } else {
            digits
        };
        let es = if exp < 0 { '-' } else { '+' };
        format!("{sign}{m}e{es}{:02}", exp.unsigned_abs())
    }
}

/// Python truthiness of a JSON value (`bool(x)`).
#[must_use]
pub fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(m) => !m.is_empty(),
    }
}

/// `d.get(k)` with `None` for a missing key (a present `null` is `Some(Null)`).
#[must_use]
pub fn get<'a>(d: &'a Value, k: &str) -> Option<&'a Value> {
    d.as_object().and_then(|m| m.get(k))
}

/// `d.get(k, default)`: the value, or `default` when the key is absent.
#[must_use]
pub fn get_or(d: &Value, k: &str, default: Value) -> Value {
    get(d, k).cloned().unwrap_or(default)
}

/// `d.get(k)` with a missing key read as `null` (Python `None`).
#[must_use]
pub fn get_null(d: &Value, k: &str) -> Value {
    get(d, k).cloned().unwrap_or(Value::Null)
}

/// `d.get(k) or []` as a slice.
#[must_use]
pub fn list_or_empty<'a>(d: &'a Value, k: &str) -> &'a [Value] {
    match get(d, k) {
        Some(Value::Array(a)) => a,
        _ => &[],
    }
}

/// A Python `int(x)` of a JSON integer (bools count as 0/1).
#[must_use]
pub fn as_int(v: &Value) -> Option<i64> {
    match v {
        Value::Bool(b) => Some(i64::from(*b)),
        Value::Number(n) => n.as_i64(),
        _ => None,
    }
}

/// An empty JSON object map.
#[must_use]
pub fn obj() -> Map<String, Value> {
    Map::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn dumps_matches_python() {
        let v = json!({"b": [1, 1.0, true, null], "a": "x\"é\n", "c": {"z": 0.5, "y": -2}});
        assert_eq!(
            dumps(&v),
            r#"{"a": "x\"\u00e9\n", "b": [1, 1.0, true, null], "c": {"y": -2, "z": 0.5}}"#
        );
    }

    #[test]
    fn float_repr_matches_python() {
        for (x, s) in [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (1e16, "1e+16"),
            (1.5e-5, "1.5e-05"),
            (0.0001, "0.0001"),
            (123_456.789, "123456.789"),
            (-2.5, "-2.5"),
            (1e15, "1000000000000000.0"),
            (f64::from(0.1f32), "0.10000000149011612"),
        ] {
            assert_eq!(py_float_repr(x), s, "{x}");
        }
    }
}
