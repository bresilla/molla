use super::*;
use molla_math::constitutive::MembraneStrainLimit;

impl BarrierShell {
    pub(super) fn strain_neighborhoods(&mut self) -> Result<()> {
        self.strain_weights = vec![0.0; self.elastic_elements.len()];
        self.strain_faces = self.elastic_elements.iter().enumerate()
            .filter_map(|(i,e)| e.strain_limit().map(|_| i)).collect();
        if self.strain_faces.is_empty() { self.strain_neighbors.clear(); return Ok(()); }
        let stencils: Vec<_> = self.elastic_elements.iter().map(|e| e.vertices().to_vec())
            .chain(self.springs.iter().map(|s| vec![s.a,s.b]))
            .chain(self.tets.iter().map(|t| t.ids.to_vec())).collect();
        let faces: Vec<_> = self.strain_faces.iter().map(|&i| {
            let ids = self.elastic_elements[i].vertices(); [ids[0],ids[1],ids[2]]
        }).collect();
        self.strain_neighbors = molla_sim::elastic_stencil_neighborhoods(self.positions.len(), &stencils, &faces)?;
        Ok(())
    }

    pub(super) fn prepare_strain(&mut self, x: &[Vec3]) -> Result<()> {
        if self.strain_faces.is_empty() { return Ok(()); }
        let reference: Vec<_> = self.elastic_elements.iter().map(|e| e.response(x,true)).collect::<Result<_>>()?;
        for (face, neighbors) in self.strain_faces.iter().zip(&self.strain_neighbors) {
            let e = &self.elastic_elements[*face];
            let ids = e.vertices();
            let p = [self.positions[ids[0] as usize],self.positions[ids[1] as usize],self.positions[ids[2] as usize]];
            let center = p[0]/3.0+p[1]/3.0+p[2]/3.0;
            let q = |id| ids.iter().position(|&v| v == id).map_or(Vec3::ZERO, |k| p[k]-center);
            let mut stiffness = 0.0;
            for &neighbor in neighbors {
                let index = neighbor as usize;
                let term = if index < self.elastic_elements.len() {
                    let element = &self.elastic_elements[index];
                    let gradients = element.shape_gradients();
                    let mut df = Mat3::ZERO;
                    for (k,&id) in element.vertices().iter().enumerate() {
                        let d = q(id); let w = gradients[k];
                        df += Mat3::from_cols(d*w.x,d*w.y,d*w.z);
                    }
                    let hd = reference[index].hessian_mul(df);
                    df.to_cols_array().into_iter().zip(hd.to_cols_array()).map(|(a,b)| a*b).sum()
                } else if index < self.elastic_elements.len()+self.springs.len() {
                    let s = self.springs[index-self.elastic_elements.len()];
                    let d = x[s.b as usize]-x[s.a as usize];
                    let length = d.length().max(1.0e-12); let n = d/length;
                    let t = (1.0-s.rest/length).max(0.0);
                    let dq = q(s.b)-q(s.a);
                    s.k*(t*dq.length_squared()+(1.0-t)*n.dot(dq).powi(2))
                } else {
                    let t = self.tets[index-self.elastic_elements.len()-self.springs.len()];
                    let g = Self::tet_volume_grads(x,t.ids);
                    t.kv*g.iter().zip(t.ids).map(|(g,id)| g.dot(q(id))).sum::<Real>().powi(2)
                };
                if !term.is_finite() { return Err(Error::Solver("barrier: nonfinite strain reference".into())); }
                stiffness += term.max(0.0);
            }
            let limit = e.strain_limit().unwrap();
            let gap = limit.maximum_stretch()? - MembraneStrainLimit::largest_stretch(e.deformation(x)?)?;
            if gap <= 0.0 { return Err(Error::Solver("barrier: invalid strain stiffness gap".into())); }
            stiffness += e.strain_mass()/(gap*gap);
            finite_scalars(&[stiffness], "strain stiffness")?;
            self.strain_weights[*face] = stiffness;
        }
        Ok(())
    }

    pub(super) fn material_response(&self, index: usize, x: &[Vec3], projected: bool) -> Result<MaterialResponse> {
        let e = &self.elastic_elements[index];
        let mut response = e.response(x,projected)?;
        let weight = self.strain_weights.get(index).copied().unwrap_or(0.0);
        if let Some(limit) = e.strain_limit() && weight > 0.0 {
            let barrier = limit.evaluate(e.deformation(x)?)?;
            response.energy += weight*barrier.energy;
            response.gradient += barrier.gradient*weight;
            for i in 0..9 { for j in 0..9 { response.hessian[i][j] += weight*barrier.hessian[i][j]; } }
        }
        if !response.energy.is_finite() || !response.gradient.is_finite() || response.hessian.iter().flatten().any(|v| !v.is_finite()) {
            return Err(Error::Solver("barrier: nonfinite strain response".into()));
        }
        Ok(response)
    }

    /// Freeze strain stiffness for energy/derivative probes at nearby configurations.
    pub fn fd_probe_prepare_strain(&mut self, x: &[Vec3]) -> Result<Vec<Real>> {
        self.strain_neighborhoods()?;
        self.prepare_strain(x)?;
        Ok(self.strain_weights.clone())
    }
}
