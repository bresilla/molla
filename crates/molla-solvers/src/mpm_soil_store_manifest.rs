//! Version-one manifest streaming and bounded file-backed entry lookup.

use super::*;
use std::io::{BufReader, BufWriter, Seek, SeekFrom};
use std::sync::Mutex;

const PREFIX_BYTES: usize = HEADER_BYTES + 16;
const BUFFER_BYTES: usize = 8192;

struct Cursor {
    reader: BufReader<File>,
    position: Option<u64>,
}

pub(super) struct ManifestFile {
    pub(super) generation: u64,
    pub(super) metadata: Vec<u8>,
    pub(super) tiles: u32,
    pub(super) particles: u32,
    pub(super) tree: Option<tree_manifest::Tree>,
    digest: [u8; 32],
    offset: u64,
    cursor: Mutex<Cursor>,
}

fn decode_entry(bytes: &[u8; ENTRY_BYTES as usize]) -> (SoilTileCoord, Entry) {
    let key = SoilTileCoord(std::array::from_fn(|a| {
        i64::from_le_bytes(bytes[8 * a..8 * a + 8].try_into().unwrap())
    }));
    let entry = Entry {
        count: u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
        hash: bytes[28..].try_into().unwrap(),
    };
    (key, entry)
}

fn encode_entry(key: SoilTileCoord, entry: &Entry) -> [u8; ENTRY_BYTES as usize] {
    let mut bytes = [0; ENTRY_BYTES as usize];
    for a in 0..3 {
        bytes[8 * a..8 * a + 8].copy_from_slice(&key.0[a].to_le_bytes());
    }
    bytes[24..28].copy_from_slice(&entry.count.to_le_bytes());
    bytes[28..].copy_from_slice(&entry.hash);
    bytes
}

impl ManifestFile {
    pub(super) fn storage_bytes(&self) -> (u64, u64) {
        (
            self.offset
                + if self.tree.is_some() {
                    0
                } else {
                    u64::from(self.tiles) * ENTRY_BYTES
                },
            self.metadata.capacity() as u64 + BUFFER_BYTES as u64,
        )
    }

    pub(super) fn open(
        path: &Path,
        limits: MpmSoilStoreLimits,
        framed: bool,
        canonical: bool,
    ) -> Result<Self> {
        if !fs::symlink_metadata(path).map_err(io_error)?.is_file() {
            return Err(invalid("store manifest is not a regular file"));
        }
        let file = File::open(path).map_err(io_error)?;
        let length = file.metadata().map_err(io_error)?.len();
        let mut reader = BufReader::with_capacity(BUFFER_BYTES, file);
        let mut header = [0; PREFIX_BYTES];
        reader.read_exact(&mut header).map_err(io_error)?;
        if &header[..8] == STORE_MAGIC && header[8..12] == 2u32.to_le_bytes() {
            return Self::open_tree(path, limits, framed, canonical, None);
        }
        if &header[..8] != STORE_MAGIC || header[8..12] != 1u32.to_le_bytes() {
            return Err(invalid("invalid store manifest header"));
        }
        let generation = u64::from_le_bytes(header[44..52].try_into().unwrap());
        let metadata_bytes = u32::from_le_bytes(header[52..56].try_into().unwrap());
        let tiles = u32::from_le_bytes(header[56..60].try_into().unwrap());
        let offset = PREFIX_BYTES as u64 + u64::from(metadata_bytes);
        if generation == 0
            || u64::from(metadata_bytes) > MAX_METADATA
            || tiles > limits.tiles
            || length != offset + u64::from(tiles) * ENTRY_BYTES
        {
            return Err(invalid("store manifest dimensions exceed bounds"));
        }
        let mut metadata = vec![0; metadata_bytes as usize];
        reader.read_exact(&mut metadata).map_err(io_error)?;
        let mut hash = blake3::Hasher::new();
        hash.update(&header[HEADER_BYTES..]);
        hash.update(&metadata);
        let mut previous = None;
        let mut particles = 0u64;
        for _ in 0..tiles {
            let mut bytes = [0; ENTRY_BYTES as usize];
            reader.read_exact(&mut bytes).map_err(io_error)?;
            hash.update(&bytes);
            let (key, entry) = decode_entry(&bytes);
            particles += u64::from(entry.count);
            if entry.count == 0
                || particles > u64::from(limits.particles)
                || previous.is_some_and(|last| last >= key)
            {
                return Err(invalid("store tile counts/order exceed bounds"));
            }
            previous = Some(key);
        }
        let mut extra = [0];
        if reader.read(&mut extra).map_err(io_error)? != 0
            || header[12..HEADER_BYTES] != *hash.finalize().as_bytes()
        {
            return Err(invalid("invalid store manifest digest or trailing bytes"));
        }
        if canonical {
            MpmSoilCanonicalTileArchive::from_bytes(&metadata, 0)?;
        } else {
            decode_archive(&metadata, 0, framed)?;
        }
        Ok(Self {
            generation,
            metadata,
            tiles,
            particles: particles as u32,
            tree: None,
            digest: header[12..HEADER_BYTES].try_into().unwrap(),
            offset,
            cursor: Mutex::new(Cursor {
                reader,
                position: Some(length),
            }),
        })
    }

    fn open_tree(
        path: &Path,
        limits: MpmSoilStoreLimits,
        framed: bool,
        canonical: bool,
        expected: Option<Option<&index::Link>>,
    ) -> Result<Self> {
        let bytes = read_bounded(path, tree_manifest::PREFIX as u64 + MAX_METADATA)?;
        let decoded = tree_manifest::decode(&bytes, limits, framed, canonical)?;
        let tree = tree_manifest::Tree {
            directory: path
                .parent()
                .ok_or_else(|| invalid("manifest parent missing"))?
                .to_path_buf(),
            root: decoded.root,
        };
        if let Some(expected) = expected {
            if tree.root.as_ref() != expected {
                return Err(invalid("created root differs from validated candidate"));
            }
        } else {
            tree.validate()?;
        }
        let (tiles, particles) = tree
            .root
            .as_ref()
            .map_or((0, 0), |root| (root.tiles, root.particles));
        Ok(Self {
            generation: decoded.generation,
            metadata: decoded.metadata,
            tiles,
            particles,
            tree: Some(tree),
            digest: bytes[12..HEADER_BYTES].try_into().unwrap(),
            offset: bytes.len() as u64,
            cursor: Mutex::new(Cursor {
                reader: BufReader::with_capacity(BUFFER_BYTES, File::open(path).map_err(io_error)?),
                position: None,
            }),
        })
    }

    pub(super) fn open_created_tree(
        path: &Path,
        limits: MpmSoilStoreLimits,
        framed: bool,
        canonical: bool,
        expected: Option<&index::Link>,
    ) -> Result<Self> {
        Self::open_tree(path, limits, framed, canonical, Some(expected))
    }

    pub(super) fn index_files(&self) -> Result<Vec<PathBuf>> {
        self.tree
            .as_ref()
            .map_or_else(|| Ok(Vec::new()), |tree| tree.files())
    }

    fn entry(&self, index: u32) -> Result<(SoilTileCoord, Entry)> {
        if index >= self.tiles {
            return Err(invalid("manifest entry index exceeds count"));
        }
        let offset = self.offset + u64::from(index) * ENTRY_BYTES;
        let mut cursor = self
            .cursor
            .lock()
            .map_err(|_| invalid("manifest cursor poisoned"))?;
        if cursor.position != Some(offset) {
            cursor.position = None;
            cursor
                .reader
                .seek(SeekFrom::Start(offset))
                .map_err(io_error)?;
        }
        cursor.position = None;
        let mut bytes = [0; ENTRY_BYTES as usize];
        cursor.reader.read_exact(&mut bytes).map_err(io_error)?;
        cursor.position = Some(offset + ENTRY_BYTES);
        Ok(decode_entry(&bytes))
    }

    pub(super) fn get(&self, key: SoilTileCoord) -> Result<Option<Entry>> {
        if let Some(tree) = &self.tree {
            return tree.get(key);
        }
        let mut lower = 0;
        let mut upper = self.tiles;
        while lower < upper {
            let middle = lower + (upper - lower) / 2;
            let (candidate, entry) = self.entry(middle)?;
            match candidate.cmp(&key) {
                std::cmp::Ordering::Less => lower = middle + 1,
                std::cmp::Ordering::Greater => upper = middle,
                std::cmp::Ordering::Equal => return Ok(Some(entry)),
            }
        }
        Ok(None)
    }

    pub(super) fn iter(&self) -> Box<dyn Iterator<Item = Result<(SoilTileCoord, Entry)>> + '_> {
        if let Some(tree) = &self.tree {
            return Box::new(tree.iter());
        }
        let mut hash = blake3::Hasher::new();
        hash.update(&self.generation.to_le_bytes());
        hash.update(&(self.metadata.len() as u32).to_le_bytes());
        hash.update(&self.tiles.to_le_bytes());
        hash.update(&self.metadata);
        let mut index = 0u64;
        Box::new(std::iter::from_fn(move || {
            if index < u64::from(self.tiles) {
                let result = self.entry(index as u32);
                index += 1;
                match &result {
                    Ok((key, entry)) => {
                        hash.update(&encode_entry(*key, entry));
                    }
                    Err(_) => index = u64::from(self.tiles) + 1,
                }
                Some(result)
            } else if index == u64::from(self.tiles) {
                index += 1;
                (hash.finalize().as_bytes() != &self.digest)
                    .then(|| Err(invalid("manifest changed during enumeration")))
            } else {
                None
            }
        }))
    }

    pub(super) fn retained(&self, replace: &BTreeSet<SoilTileCoord>) -> Result<(u32, u32)> {
        let mut tiles = self.tiles;
        let mut particles = self.particles;
        for &key in replace {
            if let Some(entry) = self.get(key)? {
                tiles = tiles
                    .checked_sub(1)
                    .ok_or_else(|| invalid("manifest tile total mismatch"))?;
                particles = particles
                    .checked_sub(entry.count)
                    .ok_or_else(|| invalid("manifest particle total mismatch"))?;
            }
        }
        Ok((tiles, particles))
    }

    #[cfg(test)]
    pub(super) fn snapshot(&self) -> Result<BTreeMap<SoilTileCoord, Entry>> {
        self.iter().collect()
    }

    #[cfg(test)]
    pub(super) fn materialize(self) -> Result<Manifest> {
        let mut entries = BTreeMap::new();
        let mut particles = 0u64;
        let mut hash = blake3::Hasher::new();
        hash.update(&self.generation.to_le_bytes());
        hash.update(&(self.metadata.len() as u32).to_le_bytes());
        hash.update(&self.tiles.to_le_bytes());
        hash.update(&self.metadata);
        for i in 0..self.tiles {
            let (key, entry) = self.entry(i)?;
            particles += u64::from(entry.count);
            hash.update(&encode_entry(key, &entry));
            entries.insert(key, entry);
        }
        if entries.len() != self.tiles as usize
            || particles != u64::from(self.particles)
            || hash.finalize().as_bytes() != &self.digest
        {
            return Err(invalid("manifest changed during materialization"));
        }
        Ok(Manifest {
            generation: self.generation,
            particles: self.particles,
            metadata: self.metadata,
            entries,
        })
    }
}

pub(super) fn merged<'a>(
    base: Option<&'a ManifestFile>,
    replace: Option<&'a BTreeSet<SoilTileCoord>>,
    entries: &'a BTreeMap<SoilTileCoord, Entry>,
) -> impl Iterator<Item = Result<(SoilTileCoord, Entry)>> + 'a {
    let mut old = base
        .into_iter()
        .flat_map(ManifestFile::iter)
        .filter(move |item| {
            item.as_ref().map_or(true, |(key, _)| {
                replace.is_some_and(|set| !set.contains(key))
            })
        })
        .peekable();
    let mut new = entries.iter().peekable();
    std::iter::from_fn(move || match (old.peek(), new.peek()) {
        (Some(Err(_)), _) => old.next(),
        (Some(Ok((old_key, _))), Some((new_key, _))) if old_key >= *new_key => {
            if old_key == *new_key {
                old.next();
                return Some(Err(invalid("manifest patch overlaps retained tile")));
            }
            new.next().map(|(&key, entry)| Ok((key, entry.clone())))
        }
        (Some(_), _) => old.next(),
        (None, _) => new.next().map(|(&key, entry)| Ok((key, entry.clone()))),
    })
}

pub(super) fn write_entries(
    file: &mut File,
    generation: u64,
    metadata: &[u8],
    tiles: u32,
    expected_particles: u32,
    entries: impl Iterator<Item = Result<(SoilTileCoord, Entry)>>,
    limits: MpmSoilStoreLimits,
) -> Result<u64> {
    if generation == 0 || metadata.len() as u64 > MAX_METADATA || tiles > limits.tiles {
        return Err(invalid("store manifest dimensions exceed bounds"));
    }
    let mut writer = BufWriter::with_capacity(BUFFER_BYTES, file);
    let mut header = [0; PREFIX_BYTES];
    header[..8].copy_from_slice(STORE_MAGIC);
    header[8..12].copy_from_slice(&1u32.to_le_bytes());
    header[44..52].copy_from_slice(&generation.to_le_bytes());
    header[52..56].copy_from_slice(&(metadata.len() as u32).to_le_bytes());
    header[56..60].copy_from_slice(&tiles.to_le_bytes());
    writer.write_all(&header).map_err(io_error)?;
    writer.write_all(metadata).map_err(io_error)?;
    let mut hash = blake3::Hasher::new();
    hash.update(&header[HEADER_BYTES..]);
    hash.update(metadata);
    let mut particles = 0u64;
    let mut count = 0u64;
    let mut previous = None;
    for item in entries {
        let (key, entry) = item?;
        count += 1;
        particles += u64::from(entry.count);
        if count > u64::from(tiles)
            || entry.count == 0
            || particles > u64::from(limits.particles)
            || previous.is_some_and(|last| last >= key)
        {
            return Err(invalid("store tile counts/order exceed bounds"));
        }
        previous = Some(key);
        let bytes = encode_entry(key, &entry);
        writer.write_all(&bytes).map_err(io_error)?;
        hash.update(&bytes);
    }
    if count != u64::from(tiles) || particles != u64::from(expected_particles) {
        return Err(invalid("manifest cached totals mismatch"));
    }
    writer.seek(SeekFrom::Start(12)).map_err(io_error)?;
    writer
        .write_all(hash.finalize().as_bytes())
        .map_err(io_error)?;
    writer.flush().map_err(io_error)?;
    Ok(PREFIX_BYTES as u64 + metadata.len() as u64 + u64::from(tiles) * ENTRY_BYTES)
}

#[cfg(test)]
impl Manifest {
    pub(super) fn write_to(&self, file: &mut File, limits: MpmSoilStoreLimits) -> Result<u64> {
        if self.generation == 0
            || self.metadata.len() as u64 > MAX_METADATA
            || self.entries.len() as u64 > u64::from(limits.tiles)
        {
            return Err(invalid("store manifest dimensions exceed bounds"));
        }
        let mut writer = BufWriter::with_capacity(BUFFER_BYTES, file);
        let mut header = [0; PREFIX_BYTES];
        header[..8].copy_from_slice(STORE_MAGIC);
        header[8..12].copy_from_slice(&1u32.to_le_bytes());
        header[44..52].copy_from_slice(&self.generation.to_le_bytes());
        header[52..56].copy_from_slice(&(self.metadata.len() as u32).to_le_bytes());
        header[56..60].copy_from_slice(&(self.entries.len() as u32).to_le_bytes());
        writer.write_all(&header).map_err(io_error)?;
        writer.write_all(&self.metadata).map_err(io_error)?;
        let mut hash = blake3::Hasher::new();
        hash.update(&header[HEADER_BYTES..]);
        hash.update(&self.metadata);
        let mut particles = 0u64;
        for (&key, entry) in &self.entries {
            particles += u64::from(entry.count);
            if entry.count == 0 || particles > u64::from(limits.particles) {
                return Err(invalid("store tile counts exceed bounds"));
            }
            let bytes = encode_entry(key, entry);
            writer.write_all(&bytes).map_err(io_error)?;
            hash.update(&bytes);
        }
        if particles != u64::from(self.particles) {
            return Err(invalid("manifest cached particle count mismatch"));
        }
        writer.seek(SeekFrom::Start(12)).map_err(io_error)?;
        writer
            .write_all(hash.finalize().as_bytes())
            .map_err(io_error)?;
        writer.flush().map_err(io_error)?;
        Ok(PREFIX_BYTES as u64
            + self.metadata.len() as u64
            + self.entries.len() as u64 * ENTRY_BYTES)
    }
}

#[cfg(test)]
#[path = "mpm_soil_store_manifest_tests.rs"]
mod tests;
