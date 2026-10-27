fn fischer_root(x:f32)->f32 {return rate_round(sqrt(x));}
fn fischer_div(a:vec2<f32>,b:vec2<f32>)->vec2<f32> {
    let high=rate_round(a.x/b.x);
    let remainder=rate_add(a,-rate_mul(high,b));
    return rate_add(vec2<f32>(high,0.0),vec2<f32>(value(remainder)/value(b),0.0));
}
fn fischer_length(n:vec3<f32>)->vec2<f32> {
    var square=vec2<f32>(0.0);
    for(var i=0u;i<3u;i++) {square=rate_add(square,rate_mul(n[i],vec2<f32>(n[i],0.0)));}
    let high=fischer_root(value(square));
    let remainder=rate_add(square,-rate_mul(high,vec2<f32>(high,0.0)));
    return rate_add(vec2<f32>(high,0.0),vec2<f32>(value(remainder)/(2.0*high),0.0));
}
fn fischer_product(a:vec2<f32>,b:vec2<f32>)->vec2<f32> {
    let high=rate_round(a.x*b.x);
    let cross=rate_round(fma(a.x,b.y,rate_round(a.y*b.x)));
    let tail=rate_round(fma(a.y,b.y,fma(a.x,b.x,-high)));
    return rate_sum(high,rate_round(tail+cross));
}
fn fischer_dot(a:array<vec2<f32>,3>,b:array<vec2<f32>,3>)->vec2<f32> {
    var sum=vec2<f32>(0.0);
    for(var i=0u;i<3u;i++) {sum=rate_add(sum,fischer_product(a[i],b[i]));}
    return sum;
}
fn fischer_residual(a:Local)->array<f32,6> {
    let lambda=fischer_dot(a.paired_normal,a.paired_k);
    let speed=fischer_dot(a.paired_normal,a.paired_relative);
    var tangent:array<vec2<f32>,3>;var slip:array<vec2<f32>,3>;var y:array<vec2<f32>,3>;
    for(var i=0u;i<3u;i++) {
        tangent[i]=rate_add(a.paired_k[i],-fischer_product(a.paired_normal[i],lambda));
        slip[i]=rate_add(a.paired_relative[i],-fischer_product(a.paired_normal[i],speed));
        y[i]=rate_add(tangent[i],-fischer_product(a.paired_rho,slip[i]));
    }
    let square=fischer_dot(y,y);let high=fischer_root(max(value(square),0.0));var radius=vec2<f32>(high,0.0);
    if high>0.0 {radius=rate_add(radius,vec2<f32>(value(rate_add(square,-rate_mul(high,radius)))/(2.0*high),0.0));}
    var cap=vec2<f32>(0.0);if value(lambda)>0.0 {cap=rate_mul(a.mu,lambda);}
    let sliding=radius.x>cap.x || (radius.x==cap.x && radius.y>cap.y);
    var ratio=vec2<f32>(0.0);if sliding {ratio=fischer_div(cap,radius);}
    let normal=fischer_precise(lambda,fischer_product(a.paired_rho,a.paired_separation))[0];
    var k=vec3<f32>(0.0);var w=a.w;
    for(var i=0u;i<3u;i++) {
        var residual=fischer_product(a.paired_rho,slip[i]);
        if sliding {residual=rate_add(tangent[i],-fischer_product(y[i],ratio));}
        k[i]=value(rate_add(fischer_product(normal,a.paired_normal[i]),residual));
        if a.wall[i]!=0.0 {w[i]=a.wall[i]*value(fischer_precise(rate_mul(a.wall[i],a.paired_w[i]),rate_mul(a.wall[i]*a.mass,a.paired_soil[i]))[0]);}
    }
    return six(k,w);
}
fn fischer(a:f32,b:f32)->vec3<f32> {
    let scale=max(abs(a),abs(b));
    if scale==0.0 {return vec3<f32>(0.0,0.2928932188,0.2928932188);}
    let x=a/scale;let y=b/scale;let r=sqrt(x*x+y*y);
    var residual=(x+y-r)*scale;
    if x+y>0.0 {residual=(2.0*x*(y/(x+y+r)))*scale;}
    var da=1.0-x/r;var db=1.0-y/r;
    if x>=0.0 {da=(y/r)*(y/(r+x));}
    if y>=0.0 {db=(x/r)*(x/(r+y));}
    return vec3<f32>(residual,da,db);
}
