
        if params.pad2!=0u {
            var diagonal=1.0/node.soil_mass.w;
            for (var j=0u;j<params.width;j++) {diagonal+=dot(rows[c*params.width+j].xyz,response[c*params.width+j].xyz);}
            if !finite(diagonal) || diagonal<=0.0 {publication_fault(2u,c+1u);continue;}
            let rho=1.0/diagonal;
            let shifted=tangent-rho*slip;let radius=length(shifted);
            var projected=shifted;
            if radius>max(cap,0.0) {projected=shifted*(max(cap,0.0)/radius);}
            let residual=normal*min(lambda,rho*strict.x)+tangent-projected;
            if !all(abs(residual)<=vec3<f32>(1e-6)) {publication_fault(8u,c+1u);}
            for (var axis=0u;axis<3u;axis++) {
                let flag=(node.boundary.x>>(2u*axis))&3u;
                let sign=select(select(0.0,1.0,flag==1u),-1.0,flag==2u);
                let residual=w[axis]-sign*max(sign*(w[axis]-node.soil_mass.w*soil[axis]),0.0);
                if !finite(residual) || abs(residual)>1e-6 {publication_fault(16u,c+1u);}
            }
        }
