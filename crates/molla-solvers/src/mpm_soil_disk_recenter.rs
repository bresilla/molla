//! Asynchronous capture and exact canonical-owner coordinate publication.

use super::*;

pub(super) struct Recenter {
    ticket: MpmSoilCheckpointTicket,
    selection: MpmSoilCanonicalSelection,
    origin: [f64; 3],
    grid: [f32; 3],
    pins: Option<[u32; 12]>,
    move_window: bool,
    cells: Option<[i32; 3]>,
}

impl MpmSoilDiskWorld {
    /// Begins an exact canonical-owner coordinate change without disk I/O.
    /// Poll with poll_region_update; stepping and other updates freeze until done.
    /// Selected archive-frame assembly occurs here; capture validation and full
    /// physical upload occur when polled. IDs, world tiles, history, remaining
    /// swept horizon and durable generation are preserved. GPU pins reuse storage.
    /// Lossy coordinates or unsupported pin offsets reject before publication.
    /// This does not move physical grid boundaries or rebase external machines.
    /// Device loss is not rollback; external coordinate consumers must refresh.
    pub fn begin_recenter(&mut self, grid_origin: [f32; 3]) -> Result<()> {
        self.begin_grid_change(grid_origin, false, None)
    }

    /// Moves the physical grid window while preserving local/world soil coordinates.
    /// Captured transfer/Jacobi support must fit both windows on the same lattice.
    /// Uses the existing asynchronous publication boundary; no particles are evicted.
    /// This does not activate new soil, move machines, or authorize later sleep.
    pub fn begin_grid_window_move(&mut self, grid_origin: [f32; 3]) -> Result<()> {
        self.begin_grid_change(grid_origin, true, None)
    }

    /// Begins an absolute integer-cell window move against the unchanged lattice anchor.
    /// Polls through the existing capture boundary; support admission precedes publication.
    /// Preserves world coordinates, particle IDs and durable archive generation.
    pub fn begin_grid_window_cells(&mut self, cells: [i32; 3]) -> Result<()> {
        self.begin_grid_change(self.simulation.cfg.grid_origin, true, Some(cells))
    }

    fn begin_grid_change(
        &mut self,
        grid_origin: [f32; 3],
        move_window: bool,
        cells: Option<[i32; 3]>,
    ) -> Result<()> {
        self.ready()?;
        if cells.is_none() && self.simulation.grid_window_cells != [0; 3] {
            return Err(invalid(
                "origin-only coordinate operations require a zero-cell window",
            ));
        }
        let (Store::Canonical { store, origin, .. }, Read::Canonical { read, .. }) =
            (self.store.as_ref().unwrap(), self.read.as_ref().unwrap())
        else {
            return Err(invalid("live recenter requires canonical disk storage"));
        };
        store.validate_read(read)?;
        let origin = if let Some(cells) = cells {
            crate::soil_candidates_gpu::validate_node_window(
                grid_origin,
                self.simulation.cfg.cell_size,
                self.simulation.cfg.grid_size,
                cells,
            )?;
            *origin
        } else if move_window {
            recenter::window_offset(self.simulation.cfg, grid_origin)?;
            *origin
        } else {
            recenter::translation(self.simulation.cfg, grid_origin)?;
            frame::translated_origin(*origin, self.simulation.cfg.grid_origin, grid_origin)?
        };
        let selection = read.select_window(origin, grid_origin, cells.unwrap_or([0; 3]))?;
        let pins = self
            .gpu_pins
            .as_ref()
            .map(|pins| pins.recentered_parameters(self.layout, &self.simulation, origin))
            .transpose()?;
        let revision = self.next_revision;
        let next = revision
            .checked_add(1)
            .ok_or_else(|| invalid("disk world revision exhausted"))?;
        let ticket = self
            .readback
            .try_capture(&self.simulation, revision)?
            .ok_or_else(|| invalid("disk capture slot unavailable"))?;
        self.pending = Some(Pending::Recenter(Recenter {
            ticket,
            selection,
            origin,
            grid: grid_origin,
            pins,
            move_window,
            cells,
        }));
        self.next_revision = next;
        Ok(())
    }

    pub(super) fn poll_recenter(&mut self, engine: &Engine, request: Recenter) -> Result<bool> {
        let completion = match self.readback.poll_next() {
            Ok(Some(completion)) => completion,
            Ok(None) => {
                self.pending = Some(Pending::Recenter(request));
                return Ok(false);
            }
            Err(error) => return Err(self.fault(error)),
        };
        if completion.ticket().revision() != request.ticket.revision()
            || !self
                .readback
                .is_compatible(&self.simulation, completion.ticket())
            || completion
                .identity()
                .is_none_or(|identity| !self.simulation.is_soil_identity_current(identity))
        {
            return Err(self.fault(invalid("disk recenter capture identity mismatch")));
        }
        let identity = completion.identity().unwrap().clone();
        let checkpoint = completion
            .into_result()
            .map_err(|error| self.fault(error))?;
        let shifted = if let Some(cells) = request.cells {
            checkpoint.with_grid_window_cells(cells)?
        } else if request.move_window {
            checkpoint.with_grid_window(request.grid)?
        } else {
            checkpoint.recentered(request.grid)?
        };
        self.simulation
            .validate_soil_restore_origin(&shifted, true)?;
        let selected = request.selection.framed_selection();
        if selected.particle_ids() != identity.particle_ids()
            || selected.material_ids() != identity.material_ids()
        {
            return Err(self.fault(invalid("disk recenter changed selection identities")));
        }
        let changed = self.simulation.cfg.grid_origin.map(f32::to_bits)
            != request.grid.map(f32::to_bits)
            || self.simulation.grid_window_cells != shifted.grid_window_cells();
        if changed {
            if !request.move_window {
                let mut encoder = self.simulation.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("canonical disk soil recenter"),
                    },
                );
                self.simulation.encode_soil_restore(&mut encoder, &shifted);
                engine.queue().unwrap().submit([encoder.finish()]);
            }
            self.simulation.cfg.grid_origin = request.grid;
            self.simulation
                .finish_identified_soil_restore(&shifted, identity);
        }
        let Store::Canonical {
            origin,
            grid,
            cells,
            ..
        } = self.store.as_mut().unwrap()
        else {
            unreachable!()
        };
        *origin = request.origin;
        *grid = request.grid;
        *cells = shifted.grid_window_cells();
        let Read::Canonical { selection, .. } = self.read.as_mut().unwrap() else {
            unreachable!()
        };
        *selection = request.selection;
        if let Some(params) = request.pins {
            self.gpu_pins.as_mut().unwrap().commit_recenter(params);
        }
        self.world_origin = Some(request.origin);
        self.captured = None;
        self.committed_revision = Some(request.ticket.revision());
        self.last_transfer = MpmSoilTransferStats {
            path: if changed && !request.move_window {
                MpmSoilTransferPath::FullUpload
            } else {
                MpmSoilTransferPath::Unchanged
            },
            retained_particles: shifted.particle_count(),
            upload_bytes: if changed && !request.move_window {
                shifted.particle_bytes() as u64
            } else {
                0
            },
            ..Default::default()
        };
        Ok(true)
    }
}
