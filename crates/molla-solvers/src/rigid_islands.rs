//! Deterministic dynamic-body groups and immovable wake dependencies.

use std::collections::{BTreeMap, BTreeSet};

use molla_core::{Error, Result};
use molla_sim::runtime::{BodyHandle, BodyKind, RigidScene};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RigidIsland {
    pub bodies: Vec<BodyHandle>,
    pub articulations: Vec<usize>,
    pub boundaries: Vec<BodyHandle>,
}

/// Owned whole-articulation snapshot, unchanged by later scene edits or steps.
/// Immovable contact/loop endpoints do not merge separate articulations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RigidIslands {
    groups: Vec<RigidIsland>,
    owners: BTreeMap<BodyHandle, usize>,
    boundary_owners: BTreeMap<BodyHandle, Vec<usize>>,
    known: BTreeSet<BodyHandle>,
}

impl RigidIslands {
    /// Group active trees, enabled loop joints and caller-supplied force/contact links.
    pub fn build(
        scene: &RigidScene,
        connections: impl IntoIterator<Item = [Option<BodyHandle>; 2]>,
    ) -> Result<Self> {
        let n = scene.model().body_count;
        let mut handles = vec![None; n];
        let mut dynamic = vec![false; n];
        let mut enabled = vec![false; n];
        let mut indices = BTreeMap::new();
        for (handle, body) in scene.bodies() {
            let index = scene
                .body_index(handle)
                .ok_or_else(|| invalid("missing island body index"))?
                .index();
            handles[index] = Some(handle);
            enabled[index] = body.enabled;
            dynamic[index] = body.enabled && body.kind == BodyKind::Dynamic;
            indices.insert(handle, index);
        }
        let handles: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.ok_or_else(|| invalid("missing island body handle")))
            .collect::<Result<_>>()?;
        let mut union = Union::new(n);
        let mut boundaries = Vec::new();
        let mut body_articulation = vec![None; n];
        let children = scene.model().joint_child.host()?;
        for (art, range) in scene
            .model()
            .articulation_start
            .host()?
            .windows(2)
            .enumerate()
        {
            let mut first = None;
            let mut anchors = Vec::new();
            for &child in &children[range[0] as usize..range[1] as usize] {
                let body = child as usize;
                if dynamic
                    .get(body)
                    .copied()
                    .ok_or_else(|| invalid("invalid island joint child"))?
                {
                    body_articulation[body] = Some(art);
                    if let Some(first) = first {
                        union.join(first, body);
                    } else {
                        first = Some(body);
                    }
                } else if enabled[body] {
                    anchors.push(body);
                }
            }
            if let Some(first) = first {
                boundaries.extend(anchors.into_iter().map(|anchor| (first, anchor)));
            }
        }
        let loops = scene.loop_joints().filter_map(|(id, joint)| {
            scene
                .joint_settings(id)
                .filter(|settings| settings.enabled)
                .map(|_| [joint.parent, Some(joint.child)])
        });
        for pair in loops.chain(connections) {
            let mut endpoints = [None; 2];
            for (slot, handle) in endpoints.iter_mut().zip(pair) {
                *slot = handle
                    .map(|handle| {
                        indices
                            .get(&handle)
                            .copied()
                            .ok_or_else(|| invalid("unknown island connection body"))
                    })
                    .transpose()?;
            }
            let [Some(a), Some(b)] = endpoints else {
                continue;
            };
            if !enabled[a] || !enabled[b] {
                continue;
            }
            match (dynamic[a], dynamic[b]) {
                (true, true) => union.join(a, b),
                (true, false) => boundaries.push((a, b)),
                (false, true) => boundaries.push((b, a)),
                (false, false) => {}
            }
        }
        let mut groups = BTreeMap::<usize, RigidIsland>::new();
        for (body, &handle) in handles
            .iter()
            .enumerate()
            .filter(|(body, _)| dynamic[*body])
        {
            let group = groups
                .entry(union.root(body))
                .or_insert_with(|| RigidIsland {
                    bodies: Vec::new(),
                    articulations: Vec::new(),
                    boundaries: Vec::new(),
                });
            group.bodies.push(handle);
            if let Some(art) = body_articulation[body] {
                group.articulations.push(art);
            }
        }
        for (body, boundary) in boundaries {
            groups
                .get_mut(&union.root(body))
                .unwrap()
                .boundaries
                .push(handles[boundary]);
        }
        let mut groups: Vec<_> = groups.into_values().collect();
        for group in &mut groups {
            group.bodies.sort_unstable();
            group.articulations.sort_unstable();
            group.articulations.dedup();
            group.boundaries.sort_unstable();
            group.boundaries.dedup();
        }
        groups.sort_by_key(|group| group.bodies[0]);
        let owners = groups
            .iter()
            .enumerate()
            .flat_map(|(index, group)| group.bodies.iter().map(move |&body| (body, index)))
            .collect();
        let mut boundary_owners = BTreeMap::<_, Vec<_>>::new();
        for (index, group) in groups.iter().enumerate() {
            for &body in &group.boundaries {
                boundary_owners.entry(body).or_default().push(index);
            }
        }
        Ok(Self {
            groups,
            owners,
            boundary_owners,
            known: indices.into_keys().collect(),
        })
    }

    pub fn groups(&self) -> &[RigidIsland] {
        &self.groups
    }

    pub fn body_island(&self, body: BodyHandle) -> Option<&RigidIsland> {
        self.owners.get(&body).map(|&index| &self.groups[index])
    }

    /// Dynamic members affected by edits to these dynamic or immovable bodies.
    pub fn affected_bodies(&self, changed: &[BodyHandle]) -> Result<Vec<BodyHandle>> {
        if changed.iter().any(|body| !self.known.contains(body)) {
            return Err(invalid("unknown island activation body"));
        }
        let mut groups = BTreeSet::new();
        for body in changed {
            if let Some(&owner) = self.owners.get(body) {
                groups.insert(owner);
            }
            if let Some(owners) = self.boundary_owners.get(body) {
                groups.extend(owners.iter().copied());
            }
        }
        let mut affected = BTreeSet::new();
        for index in groups {
            affected.extend(self.groups[index].bodies.iter().copied());
        }
        Ok(affected.into_iter().collect())
    }
}

fn invalid(message: &str) -> Error {
    Error::Solver(message.into())
}

struct Union {
    parent: Vec<usize>,
}

impl Union {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    fn root(&mut self, mut body: usize) -> usize {
        while self.parent[body] != body {
            self.parent[body] = self.parent[self.parent[body]];
            body = self.parent[body];
        }
        body
    }

    fn join(&mut self, a: usize, b: usize) {
        let (a, b) = (self.root(a), self.root(b));
        self.parent[a.max(b)] = a.min(b);
    }
}
