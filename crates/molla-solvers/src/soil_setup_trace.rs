use std::{
    cell::{Cell, RefCell},
    fs::File,
    io::Write,
    rc::Rc,
    time::Instant,
};

thread_local! {
    static CURRENT: RefCell<Option<Rc<Trace>>> = const { RefCell::new(None) };
}

struct Trace {
    file: RefCell<File>,
    start: Instant,
    next: Cell<u64>,
}

impl Trace {
    fn emit(&self, event: &str, id: u64, stage: &str) {
        let row = serde_json::json!({"event":event,"id":id,"stage":stage,
            "elapsed_ms":self.start.elapsed().as_secs_f64()*1000.0,
            "unwinding":std::thread::panicking()});
        writeln!(self.file.borrow_mut(), "{row}").expect("write soil setup trace");
        self.file
            .borrow_mut()
            .flush()
            .expect("flush soil setup trace");
    }
}

pub(crate) struct Session(Rc<Trace>);

pub(crate) fn begin(path: &std::path::Path) -> std::io::Result<Session> {
    CURRENT.with(|current| {
        if current.borrow().is_some() {
            return Err(std::io::Error::other("soil setup trace already active"));
        }
        let trace = Rc::new(Trace {
            file: RefCell::new(File::create_new(path)?),
            start: Instant::now(),
            next: Cell::new(1),
        });
        trace.emit("session_begin", 0, "session");
        *current.borrow_mut() = Some(trace.clone());
        Ok(Session(trace))
    })
}

impl Drop for Session {
    fn drop(&mut self) {
        CURRENT.with(|current| {
            current.borrow_mut().take();
        });
        self.0.emit("session_exit", 0, "session");
    }
}

pub(crate) struct Stage {
    trace: Rc<Trace>,
    id: u64,
    name: &'static str,
}

pub(crate) fn stage(name: &'static str) -> Option<Stage> {
    CURRENT.with(|current| {
        current.borrow().as_ref().map(|trace| {
            let id = trace.next.get();
            assert!(id <= 1024, "soil setup trace stage capacity");
            trace.next.set(id + 1);
            trace.emit("stage_begin", id, name);
            Stage {
                trace: trace.clone(),
                id,
                name,
            }
        })
    })
}

impl Drop for Stage {
    fn drop(&mut self) {
        self.trace.emit("stage_exit", self.id, self.name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn path() -> std::path::PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "molla-setup-{}-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ))
    }
    fn rows(path: &std::path::Path) -> Vec<serde_json::Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|v| serde_json::from_str(v).unwrap())
            .collect()
    }
    #[test]
    fn setup_trace_flushes_nested_stages_and_rejects_replacement() {
        assert!(stage("disabled").is_none());
        let file = path();
        let session = begin(&file).unwrap();
        let other = path();
        assert!(begin(&other).is_err());
        assert!(!other.exists());
        {
            let _outer = stage("outer");
            {
                let _inner = stage("inner");
            }
            let r = rows(&file);
            assert_eq!(r.len(), 4);
            assert_eq!(r[1]["id"], 1);
            assert_eq!(r[2]["id"], 2);
            assert_eq!(r[3]["event"], "stage_exit");
            assert_eq!(r[3]["id"], 2);
        }
        drop(session);
        assert!(stage("disabled").is_none());
        assert!(begin(&file).is_err());
        let r = rows(&file);
        assert_eq!(r.len(), 6);
        assert_eq!(r[4]["id"], 1);
        assert_eq!(r[5]["event"], "session_exit");
        assert!(r.iter().all(|v| v["unwinding"] == false));
        assert!(
            r.windows(2)
                .all(|v| v[0]["elapsed_ms"].as_f64().unwrap()
                    <= v[1]["elapsed_ms"].as_f64().unwrap())
        );
        std::fs::remove_file(file).unwrap();
    }
    #[test]
    fn setup_trace_records_unwind_without_claiming_success() {
        let file = path();
        assert!(
            std::panic::catch_unwind(|| {
                let _session = begin(&file).unwrap();
                let _stage = stage("failed");
                panic!("injected setup failure");
            })
            .is_err()
        );
        assert!(stage("disabled").is_none());
        let r = rows(&file);
        assert_eq!(r[2]["event"], "stage_exit");
        assert_eq!(r[2]["unwinding"], true);
        assert_eq!(r[3]["unwinding"], true);
        std::fs::remove_file(file).unwrap();
    }
}
