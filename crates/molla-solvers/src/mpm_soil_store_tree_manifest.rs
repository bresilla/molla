//! Version-two root manifests and immutable index-page installation.

use super::*;

pub(super) const PREFIX: usize = 68 + index::LINK_BYTES;

pub(super) struct Tree {
    pub directory: PathBuf,
    pub root: Option<index::Link>,
}

pub(super) struct Pages<'a> {
    pub directory: &'a Path,
    pub written_bytes: u64,
}

pub(super) fn page_path(directory: &Path, hash: &[u8; 32]) -> PathBuf {
    directory.join(format!(
        "{}.index",
        blake3::Hash::from_bytes(*hash).to_hex()
    ))
}

impl index::Pages for Pages<'_> {
    fn read(&mut self, hash: &[u8; 32], buffer: &mut [u8; index::PAGE_BYTES]) -> Result<usize> {
        let bytes = read_bounded(&page_path(self.directory, hash), index::PAGE_BYTES as u64)?;
        buffer[..bytes.len()].copy_from_slice(&bytes);
        Ok(bytes.len())
    }
    fn write(&mut self, hash: &[u8; 32], bytes: &[u8]) -> Result<()> {
        if bytes.len() > index::PAGE_BYTES || blake3::hash(bytes).as_bytes() != hash {
            return Err(invalid("invalid index page installation"));
        }
        let path = page_path(self.directory, hash);
        match fs::symlink_metadata(&path) {
            Ok(_) => {
                if read_bounded(&path, index::PAGE_BYTES as u64)? != bytes {
                    return Err(invalid("existing index page does not match its hash"));
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let pending = self.directory.join("pending");
                let mut file = File::create_new(&pending).map_err(io_error)?;
                file.write_all(bytes).map_err(io_error)?;
                file.sync_all().map_err(io_error)?;
                fs::rename(pending, path).map_err(io_error)?;
                self.written_bytes += bytes.len() as u64;
            }
            Err(error) => return Err(io_error(error)),
        }
        Ok(())
    }
}

impl Tree {
    pub(super) fn pages(&self) -> Pages<'_> {
        Pages {
            directory: &self.directory,
            written_bytes: 0,
        }
    }
    pub(super) fn get(&self, key: SoilTileCoord) -> Result<Option<Entry>> {
        index::get(&mut self.pages(), self.root.as_ref(), key)
    }
    pub(super) fn iter(&self) -> impl Iterator<Item = Result<(SoilTileCoord, Entry)>> + '_ {
        index::Iter::new(self.pages(), self.root.as_ref())
    }
    pub(super) fn validate(&self) -> Result<()> {
        index::scan(&mut self.pages(), self.root.as_ref(), |_, _| Ok(()))
    }
    pub(super) fn files(&self) -> Result<Vec<PathBuf>> {
        let mut paths = Vec::new();
        index::visit_pages(&mut self.pages(), self.root.as_ref(), |link| {
            paths.push(page_path(&self.directory, &link.hash));
            Ok(())
        })?;
        Ok(paths)
    }
}

pub(super) fn encode(
    generation: u64,
    metadata: &[u8],
    root: Option<&index::Link>,
    limits: MpmSoilStoreLimits,
) -> Result<Vec<u8>> {
    let (tiles, particles) = root.map_or((0, 0), |link| (link.tiles, link.particles));
    if generation == 0
        || metadata.len() as u64 > MAX_METADATA
        || tiles > limits.tiles
        || particles > limits.particles
        || root.is_some_and(|link| !index::valid_link(link))
    {
        return Err(invalid("incremental manifest dimensions exceed bounds"));
    }
    let mut bytes = vec![0; PREFIX + metadata.len()];
    bytes[..8].copy_from_slice(STORE_MAGIC);
    bytes[8..12].copy_from_slice(&2u32.to_le_bytes());
    bytes[44..52].copy_from_slice(&generation.to_le_bytes());
    bytes[52..56].copy_from_slice(&(metadata.len() as u32).to_le_bytes());
    bytes[56..60].copy_from_slice(&tiles.to_le_bytes());
    bytes[60..64].copy_from_slice(&particles.to_le_bytes());
    if let Some(root) = root {
        bytes[64..68].copy_from_slice(&1u32.to_le_bytes());
        index::encode_link(&mut bytes[68..PREFIX], root);
    }
    bytes[PREFIX..].copy_from_slice(metadata);
    let digest = blake3::hash(&bytes[HEADER_BYTES..]);
    bytes[12..HEADER_BYTES].copy_from_slice(digest.as_bytes());
    Ok(bytes)
}

pub(super) struct Decoded {
    pub generation: u64,
    pub metadata: Vec<u8>,
    pub root: Option<index::Link>,
}

pub(super) fn decode(
    bytes: &[u8],
    limits: MpmSoilStoreLimits,
    framed: bool,
    canonical: bool,
) -> Result<Decoded> {
    if bytes.len() < PREFIX
        || bytes.len() as u64 > PREFIX as u64 + MAX_METADATA
        || &bytes[..8] != STORE_MAGIC
        || bytes[8..12] != 2u32.to_le_bytes()
        || blake3::hash(&bytes[HEADER_BYTES..]).as_bytes() != &bytes[12..HEADER_BYTES]
    {
        return Err(invalid("invalid incremental manifest header/digest"));
    }
    let generation = u64::from_le_bytes(bytes[44..52].try_into().unwrap());
    let metadata_bytes = u32::from_le_bytes(bytes[52..56].try_into().unwrap()) as usize;
    let tiles = u32::from_le_bytes(bytes[56..60].try_into().unwrap());
    let particles = u32::from_le_bytes(bytes[60..64].try_into().unwrap());
    let flag = u32::from_le_bytes(bytes[64..68].try_into().unwrap());
    if generation == 0
        || bytes.len() != PREFIX + metadata_bytes
        || tiles > limits.tiles
        || particles > limits.particles
    {
        return Err(invalid("incremental manifest dimensions exceed bounds"));
    }
    let root = match flag {
        0 if tiles == 0 && particles == 0 && bytes[68..PREFIX].iter().all(|&v| v == 0) => None,
        1 => {
            let root = index::decode_link(&bytes[68..PREFIX]);
            if !index::valid_link(&root) || root.tiles != tiles || root.particles != particles {
                return Err(invalid("incremental manifest root totals mismatch"));
            }
            Some(root)
        }
        _ => return Err(invalid("invalid incremental manifest root flag")),
    };
    let metadata = bytes[PREFIX..].to_vec();
    if canonical {
        MpmSoilCanonicalTileArchive::from_bytes(&metadata, 0)?;
    } else {
        decode_archive(&metadata, 0, framed)?;
    }
    Ok(Decoded {
        generation,
        metadata,
        root,
    })
}
