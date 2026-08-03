// FalconDB — Postgres wire protocol server
//
// Implements enough of the Postgres frontend/backend protocol to be usable with
// psql, DBeaver, JDBC, and other clients that speak the Postgres protocol.
//
// Supported message flow:
//   Startup → AuthenticationOk → ParameterStatus × N → BackendKeyData
//            → ReadyForQuery
//   Query   → (RowDescription + DataRow × N + CommandComplete) | ErrorResponse
//            → ReadyForQuery
//   Terminate → graceful close

use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::Arc,
};

use falcondb::db::Database;

// ─── Wire protocol helpers ────────────────────────────────────────────────────

fn write_byte(w: &mut impl Write, b: u8) { w.write_all(&[b]).ok(); }
fn write_i16(w: &mut impl Write, v: i16) { w.write_all(&v.to_be_bytes()).ok(); }
fn write_i32(w: &mut impl Write, v: i32) { w.write_all(&v.to_be_bytes()).ok(); }
fn write_str(w: &mut impl Write, s: &str) { w.write_all(s.as_bytes()).ok(); write_byte(w, 0); }

/// Write a message with a 1-byte type tag and 4-byte length (length includes
/// the 4 bytes of the length field itself).
fn write_msg(buf: &mut Vec<u8>, tag: u8, body: &[u8]) {
    buf.push(tag);
    let len = (body.len() + 4) as i32;
    buf.extend_from_slice(&len.to_be_bytes());
    buf.extend_from_slice(body);
}

fn read_i32(r: &mut impl Read) -> std::io::Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_be_bytes(b))
}

fn read_bytes(r: &mut impl Read, n: usize) -> std::io::Result<Vec<u8>> {
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

// ─── Message builders ─────────────────────────────────────────────────────────

fn msg_authentication_ok() -> Vec<u8> {
    let mut body = Vec::new();
    write_i32(&mut body, 0); // AuthenticationOk code
    let mut out = Vec::new();
    write_msg(&mut out, b'R', &body);
    out
}

fn msg_parameter_status(name: &str, value: &str) -> Vec<u8> {
    let mut body = Vec::new();
    write_str(&mut body, name);
    write_str(&mut body, value);
    let mut out = Vec::new();
    write_msg(&mut out, b'S', &body);
    out
}

fn msg_backend_key_data(pid: i32, secret: i32) -> Vec<u8> {
    let mut body = Vec::new();
    write_i32(&mut body, pid);
    write_i32(&mut body, secret);
    let mut out = Vec::new();
    write_msg(&mut out, b'K', &body);
    out
}

fn msg_ready_for_query(status: u8) -> Vec<u8> {
    let mut out = Vec::new();
    write_msg(&mut out, b'Z', &[status]); // 'I' = idle, 'T' = in transaction
    out
}

fn msg_row_description(columns: &[String]) -> Vec<u8> {
    let mut body = Vec::new();
    write_i16(&mut body, columns.len() as i16);
    for col in columns {
        write_str(&mut body, col);
        write_i32(&mut body, 0);   // table OID
        write_i16(&mut body, 0);   // attribute number
        write_i32(&mut body, 25);  // data type OID: text
        write_i16(&mut body, -1);  // data type size
        write_i32(&mut body, 0);   // type modifier
        write_i16(&mut body, 0);   // format code: text
    }
    let mut out = Vec::new();
    write_msg(&mut out, b'T', &body);
    out
}

fn msg_data_row(row: &[String]) -> Vec<u8> {
    let mut body = Vec::new();
    write_i16(&mut body, row.len() as i16);
    for val in row {
        let bytes = val.as_bytes();
        write_i32(&mut body, bytes.len() as i32);
        body.extend_from_slice(bytes);
    }
    let mut out = Vec::new();
    write_msg(&mut out, b'D', &body);
    out
}

fn msg_command_complete(tag: &str) -> Vec<u8> {
    let mut body = Vec::new();
    write_str(&mut body, tag);
    let mut out = Vec::new();
    write_msg(&mut out, b'C', &body);
    out
}

fn msg_error_response(message: &str) -> Vec<u8> {
    let mut body = Vec::new();
    body.push(b'S'); write_str(&mut body, "ERROR");
    body.push(b'C'); write_str(&mut body, "42000");
    body.push(b'M'); write_str(&mut body, message);
    body.push(0);    // terminator
    let mut out = Vec::new();
    write_msg(&mut out, b'E', &body);
    out
}

// ─── Connection handler ───────────────────────────────────────────────────────

fn handle_connection(mut stream: TcpStream, db: Arc<Database>) {
    stream.set_nodelay(true).ok();

    // ── Startup message ───────────────────────────────────────────────────────
    // The startup message is special: no type byte, just length then payload.
    let Ok(len) = read_i32(&mut stream) else { return };
    let body_len = (len - 4).max(0) as usize;
    let Ok(body) = read_bytes(&mut stream, body_len) else { return };

    // Check for SSL request (protocol 1234.5679) — decline.
    if body_len >= 4 {
        let proto = i32::from_be_bytes([body[0], body[1], body[2], body[3]]);
        if proto == 80877103_i32 {
            // Send 'N' (no SSL).
            stream.write_all(b"N").ok();
            // Re-read the real startup message.
            let Ok(len2) = read_i32(&mut stream) else { return };
            let body_len2 = (len2 - 4).max(0) as usize;
            let Ok(_) = read_bytes(&mut stream, body_len2) else { return };
        }
    }

    // Send auth OK + parameter status messages.
    let mut out = Vec::new();
    out.extend(msg_authentication_ok());
    out.extend(msg_parameter_status("server_version", "14.0 (FalconDB)"));
    out.extend(msg_parameter_status("client_encoding", "UTF8"));
    out.extend(msg_parameter_status("DateStyle", "ISO, MDY"));
    out.extend(msg_backend_key_data(1, 0));
    out.extend(msg_ready_for_query(b'I'));
    if stream.write_all(&out).is_err() { return; }

    // ── Main loop ─────────────────────────────────────────────────────────────
    loop {
        let mut type_buf = [0u8; 1];
        if stream.read_exact(&mut type_buf).is_err() { break; }
        let msg_type = type_buf[0];

        let Ok(msg_len) = read_i32(&mut stream) else { break };
        let payload_len = (msg_len - 4).max(0) as usize;
        let Ok(payload) = read_bytes(&mut stream, payload_len) else { break };

        match msg_type {
            b'Q' => {
                // Simple query: null-terminated SQL string.
                let sql = std::str::from_utf8(&payload)
                    .unwrap_or("")
                    .trim_end_matches('\0')
                    .trim();

                let mut response = Vec::new();

                if sql.is_empty() {
                    response.extend(msg_command_complete("SELECT 0"));
                } else {
                    match db.sql(sql) {
                        Ok(result) => {
                            let cols = result.column_names();
                            let rows = result.to_rows();
                            let rc   = rows.len();

                            if !cols.is_empty() {
                                response.extend(msg_row_description(&cols));
                                for row in &rows {
                                    response.extend(msg_data_row(row));
                                }
                            }

                            // Determine the command tag.
                            let upper = sql.trim().to_ascii_uppercase();
                            let tag = if upper.starts_with("SELECT") || upper.starts_with("WITH") {
                                format!("SELECT {rc}")
                            } else if upper.starts_with("INSERT") {
                                format!("INSERT 0 {rc}")
                            } else if upper.starts_with("UPDATE") {
                                format!("UPDATE {rc}")
                            } else if upper.starts_with("DELETE") {
                                format!("DELETE {rc}")
                            } else {
                                "OK".to_string()
                            };
                            response.extend(msg_command_complete(&tag));
                        }
                        Err(e) => {
                            response.extend(msg_error_response(&e.to_string()));
                        }
                    }
                }
                response.extend(msg_ready_for_query(b'I'));
                if stream.write_all(&response).is_err() { break; }
            }

            b'X' => {
                // Terminate.
                break;
            }

            _ => {
                // Unknown message — send an error and keep going.
                let mut response = Vec::new();
                response.extend(msg_error_response("unsupported message type"));
                response.extend(msg_ready_for_query(b'I'));
                if stream.write_all(&response).is_err() { break; }
            }
        }
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() -> std::io::Result<()> {
    let addr = std::env::var("FALCONDB_PG_BIND")
        .unwrap_or_else(|_| "127.0.0.1:5433".into());

    let listener = TcpListener::bind(&addr)?;
    let db = Arc::new(Database::in_memory());

    eprintln!("FalconDB Postgres server listening on {addr}");
    eprintln!("Connect with:  psql -h 127.0.0.1 -p 5433 -U any falcondb");

    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let db = Arc::clone(&db);
                std::thread::spawn(move || handle_connection(s, db));
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }

    Ok(())
}
