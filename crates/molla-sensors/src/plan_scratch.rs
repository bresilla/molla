//! Per-producer planning storage kept across frames.

use std::collections::HashSet;

/// Planning vectors that are cleared and refilled for every encoded frame,
/// keeping their capacity so steady-state frames do not allocate them.
pub(crate) struct PlanScratch<D, M = ()> {
    /// GPU descriptors uploaded for the frame being planned.
    pub descriptors: Vec<D>,
    /// Pending-receipt index of every copied source channel.
    pub indices: Vec<usize>,
    /// Validated per-job metadata awaiting layout planning.
    pub metadata: Vec<M>,
    /// Sensor identities seen in the frame, for duplicate rejection.
    pub ids: HashSet<u64>,
}

impl<D, M> Default for PlanScratch<D, M> {
    fn default() -> Self {
        Self {
            descriptors: Vec::new(),
            indices: Vec::new(),
            metadata: Vec::new(),
            ids: HashSet::new(),
        }
    }
}

impl<D, M> PlanScratch<D, M> {
    /// Empties every vector for a new frame without releasing capacity.
    pub fn clear(&mut self) {
        self.descriptors.clear();
        self.indices.clear();
        self.metadata.clear();
        self.ids.clear();
    }
}
