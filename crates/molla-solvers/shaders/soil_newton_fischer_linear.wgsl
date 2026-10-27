var<private> fischer_smoothing:f32=0.0;
fn fischer_sqrt(square:vec2<f32>)->vec2<f32> {
    let high=fischer_root(max(value(square),0.0));
    if high==0.0 {return vec2<f32>(0.0);}
    let remainder=rate_add(square,-rate_mul(high,vec2<f32>(high,0.0)));
    return rate_add(vec2<f32>(high,0.0),vec2<f32>(value(remainder)/(2.0*high),0.0));
}
fn fischer_precise(a:vec2<f32>,b:vec2<f32>)->array<vec2<f32>,3> {
    let scale=max(max(abs(value(a)),abs(value(b))),fischer_smoothing);
    if scale==0.0 {return array<vec2<f32>,3>(vec2<f32>(0.0),vec2<f32>(0.2928932188,-1.210161704e-8),vec2<f32>(0.2928932188,-1.210161704e-8));}
    let x=fischer_div(a,vec2<f32>(scale,0.0));let y=fischer_div(b,vec2<f32>(scale,0.0));
    let epsilon=fischer_div(vec2<f32>(fischer_smoothing,0.0),vec2<f32>(scale,0.0));
    let square=fischer_product(epsilon,epsilon);
    let r=fischer_sqrt(rate_add(rate_add(fischer_product(x,x),fischer_product(y,y)),rate_mul(2.0,square)));
    let sum=rate_add(x,y);var residual=rate_add(sum,-r);
    if value(sum)>0.0 {residual=rate_mul(2.0,rate_add(fischer_product(x,fischer_div(y,rate_add(sum,r))),-fischer_div(square,rate_add(sum,r))));}
    var da=rate_add(vec2<f32>(1.0,0.0),-fischer_div(x,r));
    var db=rate_add(vec2<f32>(1.0,0.0),-fischer_div(y,r));
    if value(x)>=0.0 {da=rate_add(fischer_product(fischer_div(y,r),fischer_div(y,rate_add(r,x))),rate_mul(2.0,fischer_product(fischer_div(epsilon,r),fischer_div(epsilon,rate_add(r,x)))));}
    if value(y)>=0.0 {db=rate_add(fischer_product(fischer_div(x,r),fischer_div(x,rate_add(r,y))),rate_mul(2.0,fischer_product(fischer_div(epsilon,r),fischer_div(epsilon,rate_add(r,y)))));}
    return array<vec2<f32>,3>(rate_mul(scale,residual),da,db);
}
struct FischerLinear {
    normal:array<vec2<f32>,3>, unit:array<vec2<f32>,3>, wall:array<vec2<f32>,6>,
    rho:vec2<f32>, inverse_mass:vec2<f32>, da:vec2<f32>, db:vec2<f32>, ratio:vec2<f32>, dc:f32, branch:u32
}
fn fischer_linear(a:Local)->FischerLinear {
    var result:FischerLinear;result.normal=a.paired_normal;result.rho=a.paired_rho;
    result.inverse_mass=fischer_div(vec2<f32>(1.0,0.0),vec2<f32>(a.mass,0.0));
    let lambda=fischer_dot(a.paired_normal,a.paired_k);
    let speed=fischer_dot(a.paired_normal,a.paired_relative);
    let normal=fischer_precise(lambda,fischer_product(a.paired_rho,a.paired_separation));
    result.da=normal[1];result.db=normal[2];
    var y:array<vec2<f32>,3>;
    for(var i=0u;i<3u;i++) {
        let tangent=rate_add(a.paired_k[i],-fischer_product(a.paired_normal[i],lambda));
        let slip=rate_add(a.paired_relative[i],-fischer_product(a.paired_normal[i],speed));
        y[i]=rate_add(tangent,-fischer_product(a.paired_rho,slip));
        result.wall[2u*i]=vec2<f32>(1.0,0.0);
        if a.wall[i]!=0.0 {
            let wall=fischer_precise(rate_mul(a.wall[i],a.paired_w[i]),rate_mul(a.wall[i]*a.mass,a.paired_soil[i]));
            result.wall[2u*i]=wall[1];result.wall[2u*i+1u]=wall[2];
        }
    }
    let radius=fischer_sqrt(fischer_dot(y,y));var cap=vec2<f32>(0.0);
    if value(lambda)>0.0 {cap=rate_mul(a.mu,lambda);result.dc=a.mu;}
    if a.mu==0.0 {result.branch=2u;}
    else if radius.x>cap.x || (radius.x==cap.x && radius.y>cap.y) {
        result.branch=1u;result.ratio=fischer_div(cap,radius);
        for(var i=0u;i<3u;i++) {result.unit[i]=fischer_div(y[i],radius);}
    }
    return result;
}
fn fischer_block(local:Local,residual:array<f32,6>)->Block {
    let a=fischer_linear(local);let radial_normal=fischer_dot(a.unit,a.normal);
    var block:Block;block.residual=residual;
    for(var i=0u;i<3u;i++) {
        for(var j=0u;j<3u;j++) {
            let normal=fischer_product(a.normal[i],a.normal[j]);
            let tangent=rate_add(vec2<f32>(select(0.0,1.0,i==j),0.0),-normal);
            var dk=vec2<f32>(0.0);var dr=fischer_product(a.rho,tangent);
            if a.branch==2u {dk=tangent;dr=vec2<f32>(0.0);}
            if a.branch==1u {
                let radial=rate_add(a.unit[j],-fischer_product(a.normal[j],radial_normal));
                let projected=rate_add(tangent,-fischer_product(a.unit[i],radial));
                dk=rate_add(rate_add(tangent,-rate_mul(a.dc,fischer_product(a.unit[i],a.normal[j]))),-fischer_product(a.ratio,projected));
                dr=fischer_product(a.rho,fischer_product(a.ratio,projected));
            }
            dk=rate_add(dk,fischer_product(a.da,normal));
            dr=rate_add(dr,fischer_product(a.rho,fischer_product(a.db,normal)));
            let soil=fischer_product(dr,a.inverse_mass);
            block.direct[6u*i+j]=rate_add(dk,soil);block.direct[6u*i+3u+j]=soil;
            block.machine[3u*i+j]=-dr;
        }
        block.direct[6u*(3u+i)+i]=a.wall[2u*i+1u];
        block.direct[6u*(3u+i)+3u+i]=rate_add(a.wall[2u*i],a.wall[2u*i+1u]);
    }
    return block;
}
