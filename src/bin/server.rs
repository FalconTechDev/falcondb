use std::sync::Arc;

use actix_web::{
    delete, get, post, web, App, HttpResponse, HttpServer, Responder,
};
use serde::{Deserialize, Serialize};

use falcondb::db::Database;

// ─── State ────────────────────────────────────────────────────────────────────

struct AppState {
    db: Arc<Database>,
}

// ─── Request / Response types ─────────────────────────────────────────────────

#[derive(Deserialize)]
struct LoadCsvRequest {
    path:  String,
    table: String,
}

#[derive(Deserialize)]
struct LoadFdbRequest {
    path:  String,
    table: String,
}

#[derive(Deserialize)]
struct SqlRequest {
    sql: String,
}

#[derive(Deserialize)]
struct SaveRequest {
    path: String,
}

#[derive(Serialize)]
struct LoadResponse {
    table:     String,
    rows:      usize,
    message:   String,
}

#[derive(Serialize)]
struct QueryResponse {
    columns:   Vec<String>,
    rows:      Vec<Vec<String>>,
    row_count: usize,
    elapsed_ms: u64,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

fn err(msg: impl ToString) -> HttpResponse {
    HttpResponse::BadRequest().json(ErrorResponse { error: msg.to_string() })
}

// ─── Routes ───────────────────────────────────────────────────────────────────

#[get("/health")]
async fn health() -> impl Responder {
    HttpResponse::Ok().json(serde_json::json!({ "status": "ok", "engine": "falcondb" }))
}

#[get("/tables")]
async fn list_tables(state: web::Data<AppState>) -> impl Responder {
    let mut names = state.db.table_names();
    names.sort();
    HttpResponse::Ok().json(serde_json::json!({ "tables": names }))
}

#[get("/tables/{name}")]
async fn table_info(
    state: web::Data<AppState>,
    name:  web::Path<String>,
) -> impl Responder {
    match state.db.table_info(&name) {
        Some(info) => HttpResponse::Ok().json(info),
        None       => HttpResponse::NotFound().json(ErrorResponse {
            error: format!("table '{}' not found", name),
        }),
    }
}

#[delete("/tables/{name}")]
async fn drop_table(
    state: web::Data<AppState>,
    name:  web::Path<String>,
) -> impl Responder {
    if state.db.drop_table(&name) {
        HttpResponse::Ok().json(serde_json::json!({ "dropped": *name }))
    } else {
        HttpResponse::NotFound().json(ErrorResponse {
            error: format!("table '{}' not found", name),
        })
    }
}

#[post("/load/csv")]
async fn load_csv(
    state: web::Data<AppState>,
    body:  web::Json<LoadCsvRequest>,
) -> impl Responder {
    match state.db.load_csv(&body.path, &body.table) {
        Ok(rows) => HttpResponse::Ok().json(LoadResponse {
            table:   body.table.clone(),
            rows,
            message: format!("loaded {rows} rows into '{}'", body.table),
        }),
        Err(e) => err(e),
    }
}

#[post("/load/fdb")]
async fn load_fdb(
    state: web::Data<AppState>,
    body:  web::Json<LoadFdbRequest>,
) -> impl Responder {
    match state.db.load_fdb(&body.path, &body.table) {
        Ok(rows) => HttpResponse::Ok().json(LoadResponse {
            table:   body.table.clone(),
            rows,
            message: format!("loaded {rows} rows into '{}'", body.table),
        }),
        Err(e) => err(e),
    }
}

#[post("/tables/{name}/save")]
async fn save_fdb(
    state: web::Data<AppState>,
    name:  web::Path<String>,
    body:  web::Json<SaveRequest>,
) -> impl Responder {
    match state.db.save_fdb(&name, &body.path) {
        Ok(()) => HttpResponse::Ok().json(serde_json::json!({
            "saved": *name,
            "path":  body.path,
        })),
        Err(e) => err(e),
    }
}

#[post("/sql")]
async fn execute_sql(
    state: web::Data<AppState>,
    body:  web::Json<SqlRequest>,
) -> impl Responder {
    let start = std::time::Instant::now();

    match state.db.sql(&body.sql) {
        Ok(result) => {
            let elapsed_ms = start.elapsed().as_millis() as u64;
            HttpResponse::Ok().json(QueryResponse {
                columns:   result.column_names(),
                rows:      result.to_rows(),
                row_count: result.row_count(),
                elapsed_ms,
            })
        }
        Err(e) => err(e),
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let bind  = std::env::var("FALCONDB_BIND").unwrap_or_else(|_| "127.0.0.1:8080".into());
    let db    = Arc::new(Database::in_memory());
    let state = web::Data::new(AppState { db });

    println!("falcondb server listening on http://{bind}");

    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .app_data(web::JsonConfig::default().error_handler(|err, _| {
                let msg = err.to_string();
                actix_web::error::InternalError::from_response(
                    err,
                    HttpResponse::BadRequest().json(ErrorResponse { error: msg }),
                )
                .into()
            }))
            .service(health)
            .service(list_tables)
            .service(table_info)
            .service(drop_table)
            .service(load_csv)
            .service(load_fdb)
            .service(save_fdb)
            .service(execute_sql)
    })
    .bind(&bind)?
    .run()
    .await
}
