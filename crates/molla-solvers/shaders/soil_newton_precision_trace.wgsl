var<private> precision_sqrt_trace:bool=false;
fn precision_capture(a:Local) {
    for(var i=0u;i<3u;i++) {
        condensed[i]=a.paired_normal[i];
        condensed[3u+i]=a.paired_k[i];
        condensed[6u+i]=a.paired_relative[i];
        condensed[9u+i]=a.paired_w[i];
        condensed[12u+i]=a.paired_soil[i];
    }
    condensed[15]=a.paired_rho;condensed[16]=a.paired_separation;
    condensed[17]=vec2<f32>(a.mass,a.mu);condensed[18]=vec2<f32>(a.gap,a.separation);
    condensed[19]=a.wall.xy;condensed[20]=vec2<f32>(a.wall.z,a.rho);
    condensed[21]=vec2<f32>(a.lambda,0.0);
    let linear=fischer_linear(a);
    for(var i=0u;i<3u;i++) {condensed[32u+i]=linear.normal[i];condensed[35u+i]=linear.unit[i];}
    for(var i=0u;i<6u;i++) {condensed[38u+i]=linear.wall[i];}
    condensed[44]=linear.rho;condensed[45]=linear.inverse_mass;
    condensed[46]=linear.da;condensed[47]=linear.db;condensed[48]=linear.ratio;
    condensed[49]=vec2<f32>(linear.dc,f32(linear.branch));
    let na=fischer_dot(a.paired_normal,a.paired_k);
    let nb=fischer_product(a.paired_rho,a.paired_separation);
    let scale=max(max(abs(value(na)),abs(value(nb))),fischer_smoothing);
    let x=fischer_div(na,vec2<f32>(scale,0.0));
    let y=fischer_div(nb,vec2<f32>(scale,0.0));
    let epsilon=fischer_div(vec2<f32>(fischer_smoothing,0.0),vec2<f32>(scale,0.0));
    let square=fischer_product(epsilon,epsilon);
    let sum=rate_add(rate_add(fischer_product(x,x),fischer_product(y,y)),rate_mul(2.0,square));
    let high=sqrt(max(value(sum),0.0));precision_sqrt_trace=true;let radius=fischer_sqrt(sum);precision_sqrt_trace=false;
    condensed[50]=na;condensed[51]=nb;condensed[52]=vec2<f32>(scale,fischer_smoothing);
    condensed[53]=x;condensed[54]=y;condensed[55]=epsilon;condensed[56]=square;
    condensed[57]=sum;condensed[58]=vec2<f32>(high,value(sum));condensed[59]=radius;
    let yr=fischer_div(y,radius);let yx=fischer_div(y,rate_add(radius,x));
    let er=fischer_div(epsilon,radius);let ex=fischer_div(epsilon,rate_add(radius,x));
    let yp=fischer_product(yr,yx);let ep=fischer_product(er,ex);
    condensed[60]=yr;condensed[61]=yx;condensed[62]=er;condensed[63]=ex;
    condensed[64]=yp;condensed[65]=ep;condensed[66]=rate_add(yp,rate_mul(2.0,ep));
}
@compute @workgroup_size(64)
fn precision_local(@builtin(local_invocation_index) id:u32) {
    lane=id;rate_lane=id;
    if id<params.width {rates[id]=free_rates[id];}
    workgroupBarrier();
    if id==0u {nonlinear_contact(__CONTACT__u);}
}
