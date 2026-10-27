//! Immutable bounded radix pages for incremental soil index publication.

use super::*;

const MAGIC: &[u8; 8] = b"MOLLIDX1";
pub(super) const PAGE_BYTES: usize = 2048;
const LEAF_ENTRIES: usize = 32;
const HEADER: usize = 16;
pub(super) const LINK_BYTES: usize = 88;
const KEY_BITS: u16 = 192;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Link {
    pub hash: [u8; 32],
    pub first: SoilTileCoord,
    pub last: SoilTileCoord,
    pub tiles: u32,
    pub particles: u32,
}

pub(super) trait Pages {
    fn read(&mut self, hash: &[u8; 32], buffer: &mut [u8; PAGE_BYTES]) -> Result<usize>;
    fn write(&mut self, hash: &[u8; 32], bytes: &[u8]) -> Result<()>;
}

enum Node {
    Leaf(Vec<(SoilTileCoord, Entry)>),
    Branch { bit: u16, left: Link, right: Link },
}

fn key_bytes(key: SoilTileCoord) -> [u8; 24] {
    let mut bytes = [0; 24];
    for (axis, value) in key.0.into_iter().enumerate() {
        bytes[axis * 8..axis * 8 + 8].copy_from_slice(&((value as u64) ^ (1 << 63)).to_be_bytes());
    }
    bytes
}

fn bit(key: SoilTileCoord, index: u16) -> bool {
    key_bytes(key)[usize::from(index / 8)] & (0x80 >> (index % 8)) != 0
}

fn differing(first: SoilTileCoord, last: SoilTileCoord) -> Option<u16> {
    key_bytes(first)
        .into_iter()
        .zip(key_bytes(last))
        .enumerate()
        .find_map(|(i, (a, b))| (a != b).then(|| i as u16 * 8 + (a ^ b).leading_zeros() as u16))
}

fn encode_coord(bytes: &mut [u8], key: SoilTileCoord) {
    for axis in 0..3 {
        bytes[8 * axis..8 * axis + 8].copy_from_slice(&key.0[axis].to_le_bytes());
    }
}

fn decode_coord(bytes: &[u8]) -> SoilTileCoord {
    SoilTileCoord(std::array::from_fn(|axis| {
        i64::from_le_bytes(bytes[axis * 8..axis * 8 + 8].try_into().unwrap())
    }))
}

pub(super) fn encode_link(bytes: &mut [u8], link: &Link) {
    bytes[..32].copy_from_slice(&link.hash);
    encode_coord(&mut bytes[32..56], link.first);
    encode_coord(&mut bytes[56..80], link.last);
    bytes[80..84].copy_from_slice(&link.tiles.to_le_bytes());
    bytes[84..88].copy_from_slice(&link.particles.to_le_bytes());
}

pub(super) fn decode_link(bytes: &[u8]) -> Link {
    Link {
        hash: bytes[..32].try_into().unwrap(),
        first: decode_coord(&bytes[32..56]),
        last: decode_coord(&bytes[56..80]),
        tiles: u32::from_le_bytes(bytes[80..84].try_into().unwrap()),
        particles: u32::from_le_bytes(bytes[84..88].try_into().unwrap()),
    }
}

pub(super) fn valid_link(link: &Link) -> bool {
    link.tiles > 0
        && link.particles >= link.tiles
        && link.first <= link.last
        && ((link.tiles == 1) == (link.first == link.last))
}

impl Node {
    fn summary(&self, hash: [u8; 32]) -> Result<Link> {
        match self {
            Self::Leaf(entries) => {
                if entries.is_empty()
                    || entries.len() > LEAF_ENTRIES
                    || entries.windows(2).any(|pair| pair[0].0 >= pair[1].0)
                    || entries.iter().any(|(_, entry)| entry.count == 0)
                {
                    return Err(invalid("invalid index leaf order/count"));
                }
                let particles = entries.iter().try_fold(0u32, |sum, (_, entry)| {
                    sum.checked_add(entry.count)
                        .ok_or_else(|| invalid("index particle total overflow"))
                })?;
                Ok(Link {
                    hash,
                    first: entries[0].0,
                    last: entries.last().unwrap().0,
                    tiles: entries.len() as u32,
                    particles,
                })
            }
            Self::Branch {
                bit: split,
                left,
                right,
            } => {
                if *split >= KEY_BITS
                    || !valid_link(left)
                    || !valid_link(right)
                    || left.last >= right.first
                    || differing(left.first, right.last) != Some(*split)
                    || bit(left.last, *split)
                    || !bit(right.first, *split)
                {
                    return Err(invalid("invalid index branch bounds/prefix"));
                }
                Ok(Link {
                    hash,
                    first: left.first,
                    last: right.last,
                    tiles: left
                        .tiles
                        .checked_add(right.tiles)
                        .ok_or_else(|| invalid("index tile total overflow"))?,
                    particles: left
                        .particles
                        .checked_add(right.particles)
                        .ok_or_else(|| invalid("index particle total overflow"))?,
                })
            }
        }
    }

    fn encode(&self) -> Result<Vec<u8>> {
        self.summary([0; 32])?;
        let length = match self {
            Self::Leaf(entries) => HEADER + entries.len() * 60,
            Self::Branch { .. } => HEADER + 2 * LINK_BYTES,
        };
        let mut bytes = vec![0; length];
        bytes[..8].copy_from_slice(MAGIC);
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        match self {
            Self::Leaf(entries) => {
                bytes[14..16].copy_from_slice(&(entries.len() as u16).to_le_bytes());
                for ((key, entry), target) in
                    entries.iter().zip(bytes[HEADER..].chunks_exact_mut(60))
                {
                    encode_coord(&mut target[..24], *key);
                    target[24..28].copy_from_slice(&entry.count.to_le_bytes());
                    target[28..].copy_from_slice(&entry.hash);
                }
            }
            Self::Branch { bit, left, right } => {
                bytes[12] = 1;
                bytes[14..16].copy_from_slice(&bit.to_le_bytes());
                encode_link(&mut bytes[HEADER..HEADER + LINK_BYTES], left);
                encode_link(&mut bytes[HEADER + LINK_BYTES..], right);
            }
        }
        Ok(bytes)
    }

    fn decode(bytes: &[u8], expected: &Link, parent_bit: Option<u16>) -> Result<Self> {
        if bytes.len() < HEADER
            || bytes.len() > PAGE_BYTES
            || &bytes[..8] != MAGIC
            || bytes[8..12] != 1u32.to_le_bytes()
            || bytes[13] != 0
            || blake3::hash(bytes).as_bytes() != &expected.hash
        {
            return Err(invalid("invalid index page header/hash"));
        }
        let count = u16::from_le_bytes(bytes[14..16].try_into().unwrap());
        let node = match bytes[12] {
            0 if count > 0
                && count as usize <= LEAF_ENTRIES
                && bytes.len() == HEADER + count as usize * 60 =>
            {
                Self::Leaf(
                    bytes[HEADER..]
                        .chunks_exact(60)
                        .map(|entry| {
                            (
                                decode_coord(entry),
                                Entry {
                                    count: u32::from_le_bytes(entry[24..28].try_into().unwrap()),
                                    hash: entry[28..].try_into().unwrap(),
                                },
                            )
                        })
                        .collect(),
                )
            }
            1 if bytes.len() == HEADER + 2 * LINK_BYTES
                && count < KEY_BITS
                && parent_bit.is_none_or(|parent| count > parent) =>
            {
                Self::Branch {
                    bit: count,
                    left: decode_link(&bytes[HEADER..HEADER + LINK_BYTES]),
                    right: decode_link(&bytes[HEADER + LINK_BYTES..]),
                }
            }
            _ => return Err(invalid("invalid index page shape/depth")),
        };
        if node.summary(expected.hash)? != *expected {
            return Err(invalid("index child summary mismatch"));
        }
        Ok(node)
    }
}

fn read(pages: &mut impl Pages, link: &Link, parent: Option<u16>) -> Result<Node> {
    let mut buffer = [0; PAGE_BYTES];
    let size = pages.read(&link.hash, &mut buffer)?;
    if size > buffer.len() {
        return Err(invalid("index page exceeds read bound"));
    }
    Node::decode(&buffer[..size], link, parent)
}

fn install(pages: &mut impl Pages, node: Node) -> Result<Link> {
    let bytes = node.encode()?;
    let hash = *blake3::hash(&bytes).as_bytes();
    let link = node.summary(hash)?;
    pages.write(&hash, &bytes)?;
    Ok(link)
}

pub(super) fn get(
    pages: &mut impl Pages,
    root: Option<&Link>,
    key: SoilTileCoord,
) -> Result<Option<Entry>> {
    let Some(mut link) = root.cloned() else {
        return Ok(None);
    };
    let mut parent = None;
    loop {
        let node = read(pages, &link, parent)?;
        if key < link.first || key > link.last {
            return Ok(None);
        }
        match node {
            Node::Leaf(entries) => {
                return Ok(entries
                    .binary_search_by_key(&key, |(key, _)| *key)
                    .ok()
                    .map(|index| entries[index].1.clone()));
            }
            Node::Branch {
                bit: split,
                left,
                right,
            } => {
                parent = Some(split);
                link = if bit(key, split) { right } else { left };
            }
        }
    }
}

pub(super) fn build_sorted(
    pages: &mut impl Pages,
    entries: impl IntoIterator<Item = Result<(SoilTileCoord, Entry)>>,
) -> Result<Option<Link>> {
    let mut leaf: Vec<(SoilTileCoord, Entry)> = Vec::with_capacity(LEAF_ENTRIES + 1);
    let mut spine: Vec<(u16, Link)> = Vec::with_capacity(KEY_BITS as usize);
    let mut particles = 0u32;
    let mut tiles = 0u32;
    for item in entries {
        let (key, entry) = item?;
        if entry.count == 0 || leaf.last().is_some_and(|last| last.0 >= key) {
            return Err(invalid("invalid sorted index input"));
        }
        particles = particles
            .checked_add(entry.count)
            .ok_or_else(|| invalid("index particle total overflow"))?;
        tiles = tiles
            .checked_add(1)
            .ok_or_else(|| invalid("index tile total overflow"))?;
        if let Some((first, _)) = leaf.first() {
            let split = differing(*first, key).unwrap();
            if spine.last().is_some_and(|(parent, _)| *parent >= split) {
                let mut left = install(pages, Node::Leaf(std::mem::take(&mut leaf)))?;
                while spine.last().is_some_and(|(parent, _)| *parent >= split) {
                    let (bit, prior) = spine.pop().unwrap();
                    left = install(
                        pages,
                        Node::Branch {
                            bit,
                            left: prior,
                            right: left,
                        },
                    )?;
                }
                spine.push((split, left));
            }
        }
        leaf.push((key, entry));
        if leaf.len() > LEAF_ENTRIES {
            let split = differing(leaf[0].0, key).unwrap();
            let at = leaf.partition_point(|(key, _)| !bit(*key, split));
            let right = leaf.split_off(at);
            let left = install(pages, Node::Leaf(leaf))?;
            spine.push((split, left));
            leaf = right;
        }
        debug_assert!(spine.len() <= KEY_BITS as usize);
    }
    if leaf.is_empty() {
        return Ok(None);
    }
    let mut root = install(pages, Node::Leaf(leaf))?;
    while let Some((bit, left)) = spine.pop() {
        root = install(
            pages,
            Node::Branch {
                bit,
                left,
                right: root,
            },
        )?;
    }
    if root.tiles != tiles || root.particles != particles {
        return Err(invalid("sorted index totals mismatch"));
    }
    Ok(Some(root))
}

pub(super) fn update_many(
    pages: &mut impl Pages,
    root: Option<&Link>,
    edits: &[(SoilTileCoord, Option<Entry>)],
) -> Result<Option<Link>> {
    if edits.windows(2).any(|pair| pair[0].0 >= pair[1].0)
        || edits
            .iter()
            .any(|(_, entry)| entry.as_ref().is_some_and(|entry| entry.count == 0))
    {
        return Err(invalid("invalid index batch order/count"));
    }
    batch_at(pages, root, edits, None)
}

fn batch_at(
    pages: &mut impl Pages,
    root: Option<&Link>,
    edits: &[(SoilTileCoord, Option<Entry>)],
    parent: Option<u16>,
) -> Result<Option<Link>> {
    if edits.is_empty() {
        return Ok(root.cloned());
    }
    let Some(root) = root else {
        return build_sorted(
            pages,
            edits
                .iter()
                .filter_map(|(key, entry)| entry.clone().map(|entry| Ok((*key, entry)))),
        );
    };
    let node = read(pages, root, parent)?;
    batch_node(pages, root, node, edits)
}

fn batch_node(
    pages: &mut impl Pages,
    root: &Link,
    node: Node,
    edits: &[(SoilTileCoord, Option<Entry>)],
) -> Result<Option<Link>> {
    match node {
        Node::Leaf(entries) => {
            let mut merged: BTreeMap<_, _> = entries.iter().cloned().collect();
            for (key, value) in edits {
                if let Some(value) = value {
                    merged.insert(*key, value.clone());
                } else {
                    merged.remove(key);
                }
            }
            if merged
                .iter()
                .eq(entries.iter().map(|(key, entry)| (key, entry)))
            {
                return Ok(Some(root.clone()));
            }
            build_sorted(pages, merged.into_iter().map(Ok))
        }
        Node::Branch {
            bit: split,
            left,
            right,
        } => {
            let first = root.first.min(edits[0].0);
            let last = root.last.max(edits.last().unwrap().0);
            let combined = differing(first, last).unwrap();
            if combined < split {
                let at = edits.partition_point(|(key, _)| !bit(*key, combined));
                let old_right = bit(root.first, combined);
                let node = Node::Branch {
                    bit: split,
                    left,
                    right,
                };
                let (left, right) = if old_right {
                    let left = batch_at(pages, None, &edits[..at], Some(combined))?;
                    let right = if at == edits.len() {
                        Some(root.clone())
                    } else {
                        batch_node(pages, root, node, &edits[at..])?
                    };
                    (left, right)
                } else {
                    let left = if at == 0 {
                        Some(root.clone())
                    } else {
                        batch_node(pages, root, node, &edits[..at])?
                    };
                    let right = batch_at(pages, None, &edits[at..], Some(combined))?;
                    (left, right)
                };
                return batch_join(pages, combined, left, right);
            }
            let at = edits.partition_point(|(key, _)| !bit(*key, split));
            let next_left = batch_at(pages, Some(&left), &edits[..at], Some(split))?;
            let next_right = batch_at(pages, Some(&right), &edits[at..], Some(split))?;
            if next_left.as_ref() == Some(&left) && next_right.as_ref() == Some(&right) {
                return Ok(Some(root.clone()));
            }
            batch_join(pages, split, next_left, next_right)
        }
    }
}

fn batch_join(
    pages: &mut impl Pages,
    bit: u16,
    left: Option<Link>,
    right: Option<Link>,
) -> Result<Option<Link>> {
    match (left, right) {
        (Some(left), Some(right)) => install(pages, Node::Branch { bit, left, right }).map(Some),
        (left, right) => Ok(left.or(right)),
    }
}

#[cfg(test)]
pub(super) fn update(
    pages: &mut impl Pages,
    root: Option<&Link>,
    key: SoilTileCoord,
    value: Option<Entry>,
) -> Result<Option<Link>> {
    if value.as_ref().is_some_and(|entry| entry.count == 0) {
        return Err(invalid("zero-count index update"));
    }
    update_at(pages, root, key, value, None)
}

#[cfg(test)]
fn update_at(
    pages: &mut impl Pages,
    root: Option<&Link>,
    key: SoilTileCoord,
    value: Option<Entry>,
    parent: Option<u16>,
) -> Result<Option<Link>> {
    let Some(root) = root else {
        return value
            .map(|entry| install(pages, Node::Leaf(vec![(key, entry)])))
            .transpose();
    };
    match read(pages, root, parent)? {
        Node::Leaf(mut entries) => {
            match (entries.binary_search_by_key(&key, |(key, _)| *key), value) {
                (Ok(index), Some(entry)) => {
                    if entries[index].1 == entry {
                        return Ok(Some(root.clone()));
                    }
                    entries[index].1 = entry;
                }
                (Ok(index), None) => {
                    entries.remove(index);
                }
                (Err(index), Some(entry)) => entries.insert(index, (key, entry)),
                (Err(_), None) => return Ok(Some(root.clone())),
            }
            if entries.is_empty() {
                return Ok(None);
            }
            if entries.len() <= LEAF_ENTRIES {
                return install(pages, Node::Leaf(entries)).map(Some);
            }
            let split = differing(entries[0].0, entries.last().unwrap().0).unwrap();
            let at = entries.partition_point(|(key, _)| !bit(*key, split));
            let right = entries.split_off(at);
            let left = install(pages, Node::Leaf(entries))?;
            let right = install(pages, Node::Leaf(right))?;
            install(
                pages,
                Node::Branch {
                    bit: split,
                    left,
                    right,
                },
            )
            .map(Some)
        }
        Node::Branch {
            bit: split,
            left,
            right,
        } => {
            let outside = differing(key, root.first).is_some_and(|difference| difference < split);
            if outside {
                let Some(entry) = value else {
                    return Ok(Some(root.clone()));
                };
                let new = install(pages, Node::Leaf(vec![(key, entry)]))?;
                let split = differing(key, root.first).unwrap();
                let (left, right) = if key < root.first {
                    (new, root.clone())
                } else {
                    (root.clone(), new)
                };
                return install(
                    pages,
                    Node::Branch {
                        bit: split,
                        left,
                        right,
                    },
                )
                .map(Some);
            }
            let goes_right = bit(key, split);
            let old = if goes_right { &right } else { &left };
            let changed = update_at(pages, Some(old), key, value, Some(split))?;
            if changed.as_ref() == Some(old) {
                return Ok(Some(root.clone()));
            }
            let Some(changed) = changed else {
                return Ok(Some(if goes_right { left } else { right }));
            };
            let (left, right) = if goes_right {
                (left, changed)
            } else {
                (changed, right)
            };
            install(
                pages,
                Node::Branch {
                    bit: split,
                    left,
                    right,
                },
            )
            .map(Some)
        }
    }
}

pub(super) fn scan(
    pages: &mut impl Pages,
    root: Option<&Link>,
    mut visit: impl FnMut(SoilTileCoord, &Entry) -> Result<()>,
) -> Result<()> {
    fn walk(
        pages: &mut impl Pages,
        link: &Link,
        parent: Option<u16>,
        visit: &mut impl FnMut(SoilTileCoord, &Entry) -> Result<()>,
    ) -> Result<()> {
        match read(pages, link, parent)? {
            Node::Leaf(entries) => {
                for (key, entry) in entries {
                    visit(key, &entry)?;
                }
                Ok(())
            }
            Node::Branch { bit, left, right } => {
                walk(pages, &left, Some(bit), visit)?;
                walk(pages, &right, Some(bit), visit)
            }
        }
    }
    if let Some(root) = root {
        walk(pages, root, None, &mut visit)?;
    }
    Ok(())
}

pub(super) struct Iter<P> {
    pages: P,
    stack: Vec<(Link, Option<u16>)>,
    leaf: std::vec::IntoIter<(SoilTileCoord, Entry)>,
}

impl<P: Pages> Iter<P> {
    pub(super) fn new(pages: P, root: Option<&Link>) -> Self {
        Self {
            pages,
            stack: root
                .cloned()
                .map(|root| vec![(root, None)])
                .unwrap_or_default(),
            leaf: Vec::new().into_iter(),
        }
    }
}

impl<P: Pages> Iterator for Iter<P> {
    type Item = Result<(SoilTileCoord, Entry)>;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(entry) = self.leaf.next() {
                return Some(Ok(entry));
            }
            let (link, parent) = self.stack.pop()?;
            match read(&mut self.pages, &link, parent) {
                Ok(Node::Leaf(entries)) => self.leaf = entries.into_iter(),
                Ok(Node::Branch { bit, left, right }) => {
                    self.stack.push((right, Some(bit)));
                    self.stack.push((left, Some(bit)));
                }
                Err(error) => {
                    self.stack.clear();
                    return Some(Err(error));
                }
            }
        }
    }
}

pub(super) fn visit_pages(
    pages: &mut impl Pages,
    root: Option<&Link>,
    mut visit: impl FnMut(&Link) -> Result<()>,
) -> Result<()> {
    fn walk(
        pages: &mut impl Pages,
        link: &Link,
        parent: Option<u16>,
        visit: &mut impl FnMut(&Link) -> Result<()>,
    ) -> Result<()> {
        let node = read(pages, link, parent)?;
        visit(link)?;
        if let Node::Branch { bit, left, right } = node {
            walk(pages, &left, Some(bit), visit)?;
            walk(pages, &right, Some(bit), visit)?;
        }
        Ok(())
    }
    if let Some(root) = root {
        walk(pages, root, None, &mut visit)?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "mpm_soil_store_index_tests.rs"]
mod tests;
