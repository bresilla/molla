use std::io;
use std::process::{Child, ExitStatus};
use std::time::{Duration, Instant};

#[derive(Debug)]
pub enum WorkerExit {
    Completed(ExitStatus),
    TimedOut(ExitStatus),
}

pub fn wait_for_exit(child: &mut Child, timeout: Duration) -> io::Result<WorkerExit> {
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(WorkerExit::Completed(status)),
            Ok(None) => {}
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error);
            }
        }
        let remaining = timeout.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            if let Err(error) = child.kill() {
                if let Some(status) = child.try_wait()? {
                    return Ok(WorkerExit::Completed(status));
                }
                return Err(error);
            }
            return child.wait().map(WorkerExit::TimedOut);
        }
        std::thread::sleep(remaining.min(Duration::from_millis(100)));
    }
}
