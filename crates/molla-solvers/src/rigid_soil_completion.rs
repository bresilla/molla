use super::*;
use molla_compute::DeviceHealth;
use std::time::{Duration, Instant};

pub(super) struct CompletionDeadline<'a> {
    health: &'a DeviceHealth,
    deadline: Instant,
}

impl<'a> CompletionDeadline<'a> {
    pub(super) fn new(health: &'a DeviceHealth, timeout: Duration) -> Result<Self> {
        let deadline = Instant::now()
            .checked_add(timeout)
            .filter(|_| !timeout.is_zero())
            .ok_or_else(|| Error::Gpu("invalid soil GPU completion timeout".into()))?;
        health.check()?;
        Ok(Self { health, deadline })
    }

    fn fail(&self, message: impl std::fmt::Display) -> Error {
        let message = format!("soil GPU completion failed; logical device quarantined: {message}");
        self.health
            .record(wgpu::DeviceLostReason::Unknown, message.clone());
        Error::Gpu(message)
    }

    fn remaining(&self) -> Result<Duration> {
        self.health.check()?;
        self.deadline
            .checked_duration_since(Instant::now())
            .filter(|duration| !duration.is_zero())
            .ok_or_else(|| self.fail("completion deadline exceeded"))
    }

    pub(super) fn poll(&self, device: &wgpu::Device) -> Result<()> {
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(self.remaining()?),
            })
            .map_err(|error| self.fail(error))?;
        self.health.check()
    }

    pub(super) fn wait<T>(
        &self,
        device: &wgpu::Device,
        future: impl std::future::Future<Output = T>,
    ) -> Result<T> {
        self.wait_with(future, || self.poll(device))
    }

    fn wait_with<T>(
        &self,
        future: impl std::future::Future<Output = T>,
        mut progress: impl FnMut() -> Result<()>,
    ) -> Result<T> {
        self.remaining()?;
        let value = super::wait_with_progress(future, || {
            self.remaining()?;
            progress().map_err(|error| self.fail(error))
        })?;
        self.remaining()?;
        Ok(value)
    }

    pub(super) fn receive<T>(&self, receiver: &std::sync::mpsc::Receiver<T>) -> Result<T> {
        receiver
            .recv_timeout(self.remaining()?)
            .map_err(|error| self.fail(error))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expired_completion_quarantines_shared_health_without_polling() {
        let health = Arc::new(DeviceHealth::default());
        let observer = health.clone();
        let mut deadline = CompletionDeadline::new(&health, Duration::from_secs(5)).unwrap();
        deadline.deadline = Instant::now() - Duration::from_secs(1);
        assert!(
            deadline
                .wait_with(std::future::pending::<()>(), || panic!(
                    "expired deadline polled GPU"
                ))
                .is_err()
        );
        assert!(observer.check().is_err());
        assert!(health.loss().unwrap().message.contains("deadline exceeded"));
    }

    #[test]
    fn completion_progress_failure_is_sticky_and_success_stays_healthy() {
        let health = DeviceHealth::default();
        let deadline = CompletionDeadline::new(&health, Duration::from_secs(5)).unwrap();
        assert_eq!(
            deadline
                .wait_with(std::future::ready(42), || panic!("ready future polled GPU"))
                .unwrap(),
            42
        );
        assert!(health.check().is_ok());
        assert!(
            deadline
                .wait_with(std::future::pending::<()>(), || Err(Error::Gpu(
                    "synthetic fence timeout".into()
                )))
                .is_err()
        );
        let first = health.loss().unwrap().message;
        assert!(first.contains("synthetic fence timeout"));
        assert!(
            deadline
                .wait_with(std::future::ready(42), || Ok(()))
                .is_err()
        );
        assert_eq!(health.loss().unwrap().message, first);
    }

    #[test]
    fn missing_mapping_callback_and_invalid_budget_are_bounded() {
        let health = DeviceHealth::default();
        assert!(CompletionDeadline::new(&health, Duration::ZERO).is_err());
        assert!(health.check().is_ok());
        let deadline = CompletionDeadline::new(&health, Duration::from_millis(1)).unwrap();
        let (_sender, receiver) = std::sync::mpsc::channel::<()>();
        assert!(deadline.receive(&receiver).is_err());
        assert!(health.check().is_err());
    }

    #[test]
    fn pending_validation_cannot_extend_the_deadline() {
        let health = DeviceHealth::default();
        let deadline = CompletionDeadline::new(&health, Duration::from_millis(1)).unwrap();
        assert!(
            deadline
                .wait_with(std::future::pending::<()>(), || Ok(()))
                .is_err()
        );
        assert!(health.loss().unwrap().message.contains("deadline exceeded"));
    }
}
