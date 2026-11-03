// Reel Studio shell: the window, settings, project files, and a bridge that runs the Python engine and streams its
// JSON-line events to the UI ("engine" event: {job, line}).
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde_json::{json, Value};
use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use tauri::{AppHandle, Emitter, Manager};

fn settings_path(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(app.path().app_config_dir().map_err(|e| e.to_string())?.join("settings.json"))
}

fn read_settings(app: &AppHandle) -> Value {
    settings_path(app)
        .ok()
        .and_then(|p| fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| json!({}))
}

#[tauri::command]
fn load_settings(app: AppHandle) -> Value {
    read_settings(&app)
}

#[tauri::command]
fn save_settings(app: AppHandle, settings: Value) -> Result<(), String> {
    let p = settings_path(&app)?;
    fs::create_dir_all(p.parent().unwrap()).map_err(|e| e.to_string())?;
    fs::write(p, serde_json::to_string_pretty(&settings).unwrap()).map_err(|e| e.to_string())
}

/// the engine shipped inside the app (Resources/engine), if this is a packaged build
fn bundled_engine(app: &AppHandle) -> Option<PathBuf> {
    let d = app.path().resource_dir().ok()?.join("engine");
    d.join("app").join("engine.py").exists().then_some(d)
}

fn bundled_python(engine: &PathBuf) -> PathBuf {
    if cfg!(windows) { engine.join("python").join("python.exe") } else { engine.join("python").join("bin").join("python3") }
}

/// default paths: the bundled engine when packaged, else the development checkout
#[tauri::command]
fn engine_defaults(app: AppHandle) -> Result<Value, String> {
    let home = app.path().home_dir().map_err(|e| e.to_string())?;
    let videos = app.path().video_dir().unwrap_or_else(|_| home.join("Videos"));
    let projects = videos.join("Reel Studio").to_string_lossy().to_string();
    if let Some(e) = bundled_engine(&app) {
        return Ok(json!({ "bundled": true, "python": bundled_python(&e), "engine": e.join("app").join("engine.py"), "projects_dir": projects }));
    }
    // development build: the engine next to this checkout; python from Settings (or python3 on PATH)
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    let py = if cfg!(windows) { "python" } else { "python3" };
    Ok(json!({ "bundled": false, "python": py, "engine": repo.join("engine").join("engine.py"), "projects_dir": projects }))
}

#[tauri::command]
fn home_dir(app: AppHandle) -> Result<String, String> {
    Ok(app.path().home_dir().map_err(|e| e.to_string())?.to_string_lossy().into())
}

#[tauri::command]
fn read_json(path: String) -> Result<Value, String> {
    let s = fs::read_to_string(&path).map_err(|e| e.to_string())?;
    serde_json::from_str(&s).map_err(|e| e.to_string())
}

#[tauri::command]
fn write_json(path: String, value: Value) -> Result<(), String> {
    if let Some(dir) = PathBuf::from(&path).parent() {
        fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let tmp = format!("{path}.tmp");
    fs::write(&tmp, serde_json::to_string_pretty(&value).unwrap()).map_err(|e| e.to_string())?;
    fs::rename(&tmp, &path).map_err(|e| e.to_string())
}

#[tauri::command]
fn path_exists(path: String) -> bool {
    PathBuf::from(path).exists()
}

/// every sub-folder of `dir` that holds a project.json, newest first
#[tauri::command]
fn list_projects(dir: String) -> Vec<Value> {
    let mut out: Vec<(std::time::SystemTime, Value)> = vec![];
    if let Ok(rd) = fs::read_dir(&dir) {
        for e in rd.flatten() {
            let pj = e.path().join("project.json");
            if let (Ok(s), Ok(meta)) = (fs::read_to_string(&pj), fs::metadata(&pj)) {
                if let Ok(mut v) = serde_json::from_str::<Value>(&s) {
                    v["dir"] = json!(e.path().to_string_lossy());
                    out.push((meta.modified().unwrap_or(std::time::UNIX_EPOCH), v));
                }
            }
        }
    }
    out.sort_by(|a, b| b.0.cmp(&a.0));
    out.into_iter().map(|x| x.1).collect()
}

#[tauri::command]
fn reveal(path: String) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    let r = Command::new("open").args(["-R", &path]).spawn();
    #[cfg(target_os = "windows")]
    let r = Command::new("explorer").arg(format!("/select,{path}")).spawn();
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let r = Command::new("xdg-open").arg(PathBuf::from(&path).parent().unwrap()).spawn();
    r.map(|_| ()).map_err(|e| e.to_string())
}

/// run `python engine.py <args>`; each stdout line is forwarded as an "engine" event; returns the exit code
#[tauri::command]
async fn run_engine(app: AppHandle, job: String, args: Vec<String>) -> Result<i32, String> {
    let s = read_settings(&app);
    let get = |k: &str| s.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
    let python = get("python");
    let engine = get("engine");
    if python.is_empty() || engine.is_empty() {
        return Err("Set the engine paths in Settings first.".into());
    }
    let sep = if cfg!(windows) { ";" } else { ":" };
    // apps opened from Finder don't inherit the shell PATH: make Homebrew's ffmpeg visible (development builds)
    let path = if cfg!(windows) { std::env::var("PATH").unwrap_or_default() }
               else { format!("/opt/homebrew/bin:/usr/local/bin{sep}{}", std::env::var("PATH").unwrap_or_default()) };
    let mut cmd = Command::new(&python);
    cmd.arg("-u").arg(&engine).args(&args)
        .env("PATH", path)
        .env("PYTHONIOENCODING", "utf-8")
        .env("RS_PROVIDER", get("provider"))
        .env("RS_API_KEY", get("api_key"))
        .env("RS_MODEL", get("model"))
        .env("RS_BASE_URL", get("base_url"))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(e) = bundled_engine(&app) {
        // packaged: own ffmpeg, models in the user's app-data folder (downloaded on first run)
        let models = app.path().app_data_dir().map_err(|e| e.to_string())?.join("models");
        let _ = fs::create_dir_all(&models);
        cmd.env("RS_BIN", e.join("bin")).env("RS_MODELS", &models).env("RS_WHISPER_DIR", models.join("whisper"))
           .env("PYTHONNOUSERSITE", "1").env("PYTHONDONTWRITEBYTECODE", "1");
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW: no console flashing up
    }
    let app2 = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let mut child = cmd.spawn().map_err(|e| format!("Could not start the engine: {e}"))?;
        let mut stderr = child.stderr.take().unwrap();
        let err_thread = std::thread::spawn(move || {
            let mut buf = String::new();
            let _ = stderr.read_to_string(&mut buf);
            buf
        });
        for line in BufReader::new(child.stdout.take().unwrap()).lines().map_while(Result::ok) {
            let _ = app2.emit("engine", json!({ "job": job, "line": line }));
        }
        let status = child.wait().map_err(|e| e.to_string())?;
        let err = err_thread.join().unwrap_or_default();
        let code = status.code().unwrap_or(-1);
        if code != 0 {
            let tail: String = err.lines().filter(|l| !l.contains("objc[")).collect::<Vec<_>>().join("\n");
            let tail = tail.chars().rev().take(1200).collect::<String>().chars().rev().collect::<String>();
            let _ = app2.emit("engine", json!({ "job": job, "stderr": tail }));
        }
        Ok(code)
    })
    .await
    .map_err(|e| e.to_string())?
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            load_settings, save_settings, home_dir, engine_defaults, read_json, write_json, path_exists, list_projects, reveal, run_engine
        ])
        .run(tauri::generate_context!())
        .expect("error while running Reel Studio");
}
