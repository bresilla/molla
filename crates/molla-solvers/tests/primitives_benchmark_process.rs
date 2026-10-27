#[path = "../benches/primitives/process.rs"]
mod process;

use process::{WorkerExit, wait_for_exit};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

fn child(mode: &str) -> Child {
    Command::new(std::env::current_exe().unwrap())
        .args(["--ignored", "--exact", "worker_fixture", "--nocapture"])
        .env("MOLLA_BENCH_PROCESS_FIXTURE", mode)
        .stdout(Stdio::null())
        .spawn()
        .unwrap()
}

#[test]
#[ignore = "subprocess fixture invoked by the supervisor tests"]
fn worker_fixture() {
    match std::env::var("MOLLA_BENCH_PROCESS_FIXTURE").as_deref() {
        Ok("wait") => std::thread::sleep(Duration::from_secs(60)),
        Ok("failure") => std::process::exit(7),
        _ => {}
    }
}

#[test]
fn completed_workers_preserve_success_and_failure_status() {
    for (mode, expected) in [("success", 0), ("failure", 7)] {
        let mut child = child(mode);
        let WorkerExit::Completed(status) =
            wait_for_exit(&mut child, Duration::from_secs(10)).unwrap()
        else {
            panic!("unexpected worker timeout");
        };
        assert_eq!(status.code(), Some(expected));
        assert_eq!(child.try_wait().unwrap(), Some(status));
    }
}

#[test]
fn timed_out_worker_is_terminated_and_reaped() {
    let mut child = child("wait");
    let WorkerExit::TimedOut(status) =
        wait_for_exit(&mut child, Duration::from_millis(100)).unwrap()
    else {
        panic!("sleeping worker completed without timeout");
    };
    assert!(!status.success());
    assert_eq!(child.try_wait().unwrap(), Some(status));
}
