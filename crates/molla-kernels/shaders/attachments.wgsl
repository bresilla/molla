struct Coeff { spring: vec4<f32>, surface: vec4<f32>, petiole: vec4<f32> }
struct State { position: vec4<f32>, velocity: vec4<f32>, force: vec4<f32>, air_force: vec4<f32>, air_torque: vec4<f32>, direction: vec4<f32>, normal: vec4<f32>, angular: vec4<f32> }
struct BladeFrame { direction: vec4<f32>, normal: vec4<f32>, angular: vec4<f32> }
struct Anchor { position: vec4<f32>, velocity: vec4<f32> }
struct Params { gravity: vec4<f32>, wind: vec4<f32>, settings: vec4<f32>, coupling: vec4<f32> }
@group(0) @binding(0) var<storage,read> coefficients: array<Coeff>;
@group(0) @binding(1) var<storage,read_write> states: array<State>;
@group(0) @binding(2) var<storage,read> anchors: array<Anchor>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage,read> local_wind: array<vec4<f32>>;
@group(0) @binding(5) var<storage,read> blade_frames: array<BladeFrame>;
@group(1) @binding(0) var<uniform> cfg: Settings;
@group(1) @binding(1) var<storage,read> velocity: array<vec4<f32>>;
@group(1) @binding(2) var<storage,read_write> reactions: array<atomic<i32>>;
@group(1) @binding(3) var<storage,read_write> boundary: array<atomic<i32>>;

fn inverse_inertia(p: State, mass: f32, torque: vec3<f32>) -> vec3<f32> {
    let d=p.direction.xyz; let n=p.normal.xyz; let s=cross(d,n);
    let ix=mass*p.normal.w*p.normal.w/16.0;
    let iz=mass*p.direction.w*p.direction.w/16.0;
    return d*(dot(torque,d)/ix)+s*(dot(torque,s)/iz)+n*(dot(torque,n)/(ix+iz));
}
struct Flight { velocity: vec3<f32>, omega: vec3<f32>, direction: vec3<f32>, normal: vec3<f32> }
fn shift(f: Flight, k: Flight, dt: f32) -> Flight {
    return Flight(f.velocity+k.velocity*dt, f.omega+k.omega*dt, f.direction+k.direction*dt, f.normal+k.normal*dt);
}
fn angular_momentum(p: State, mass: f32) -> vec3<f32> {
    let d=p.direction.xyz; let n=p.normal.xyz; let s=cross(d,n);
    let ix=mass*p.normal.w*p.normal.w/16.0; let iz=mass*p.direction.w*p.direction.w/16.0;
    return d*(ix*dot(p.angular.xyz,d))+s*(iz*dot(p.angular.xyz,s))+n*((ix+iz)*dot(p.angular.xyz,n));
}
// APW (2005b), equations 3.1–3.9, with a three-dimensional circulation vector.
fn derivative(f: Flight, p: State, mass: f32, wind: vec3<f32>) -> Flight {
    let d=normalize(f.direction); let n=normalize(f.normal-d*dot(f.normal,d)); let s=cross(d,n);
    let u=f.velocity-wind; let un=dot(u,n); let speed=max(length(u),1e-12);
    let rho_area=1.225*p.angular.w;
    let added=3.141592653589793*rho_area*p.direction.w/4.0;
    let circulation= -1.2*rho_area*un/speed*cross(u,n)
        +0.5*3.141592653589793*rho_area*(s*(p.direction.w*dot(f.omega,s))+d*(p.normal.w*dot(f.omega,d)));
    let pressure_arm=n*(0.5*p.air_torque.w);
    let surface_velocity=u+cross(f.omega,pressure_arm);
    let surface_speed=max(length(surface_velocity),1e-12);
    let surface_normal=dot(surface_velocity,n);
    let drag=0.5*rho_area*(0.4+2.0*surface_normal*surface_normal/(surface_speed*surface_speed))*surface_speed;
    let drag_force=-drag*surface_velocity;
    let an=n*(added*un);
    let force=params.gravity.xyz*mass+cross(circulation,u)+drag_force-cross(f.omega,an)+n*(added*dot(cross(f.omega,u),n));
    let acceleration=(force-n*dot(force,n))/mass+n*(dot(force,n)/(mass+added));
    let ix=mass*p.normal.w*p.normal.w/16.0; let iz=mass*p.direction.w*p.direction.w/16.0;
    let jx=ix+added*p.normal.w*p.normal.w/32.0; let jz=iz+added*p.direction.w*p.direction.w/32.0; let jn=ix+iz;
    let momentum=d*(jx*dot(f.omega,d))+s*(jz*dot(f.omega,s))+n*(jn*dot(f.omega,n));
    let terminal=sqrt(2.0*mass*9.81/(3.141592653589793*rho_area));
    let axes=array<vec3<f32>,3>(d,s,n); let chords=array<f32,3>(p.normal.w,p.direction.w,p.normal.w);
    var damping=vec3<f32>(0.0);
    for (var i=0u;i<3u;i++) {
        let spin=dot(f.omega,axes[i]); let chord=chords[i];
        let coefficient=added*chord*chord*0.05*(2.0*terminal/chord+abs(spin));
        damping+=axes[i]*(coefficient*spin);
    }
    let torque=cross(an,u)-damping-cross(f.omega,momentum)
        +cross(pressure_arm,drag_force);
    return Flight(acceleration,d*(dot(torque,d)/jx)+s*(dot(torque,s)/jz)+n*(dot(torque,n)/jn),cross(f.omega,f.direction),cross(f.omega,f.normal));
}
fn fly_leaf(initial: State, a: Coeff, wind: vec3<f32>) -> State {
    var p=initial;
    let mass=a.spring.x;
    let rate=max(480.0,32.0*length(p.velocity.xyz-wind)/min(p.direction.w,p.normal.w)+16.0*length(p.angular.xyz));
    let steps=u32(max(1.0,ceil(params.settings.x*rate)));
    let dt=params.settings.x/f32(steps);
    var air_impulse=vec3<f32>(0.0); var air_moment=vec3<f32>(0.0);
    for (var step=0u;step<steps;step++) {
        let old_velocity=p.velocity.xyz; let old_momentum=angular_momentum(p,mass);
        let f=Flight(p.velocity.xyz,p.angular.xyz,p.direction.xyz,p.normal.xyz);
        let k1=derivative(f,p,mass,wind); let f2=shift(f,k1,dt*0.5);
        let k2=derivative(f2,p,mass,wind); let f3=shift(f,k2,dt*0.5);
        let k3=derivative(f3,p,mass,wind); let f4=shift(f,k3,dt);
        let k4=derivative(f4,p,mass,wind);
        let next=shift(shift(shift(shift(f,k1,dt/6.0),k2,dt/3.0),k3,dt/3.0),k4,dt/6.0);
        p.position=vec4<f32>(p.position.xyz+(f.velocity+2.0*f2.velocity+2.0*f3.velocity+f4.velocity)*(dt/6.0),p.position.w);
        p.velocity=vec4<f32>(next.velocity,p.velocity.w);
        let direction=normalize(next.direction);
        p.direction=vec4<f32>(direction,p.direction.w);
        p.normal=vec4<f32>(normalize(next.normal-direction*dot(next.normal,direction)),p.normal.w);
        p.angular=vec4<f32>(next.omega,p.angular.w);
        air_impulse+=mass*(p.velocity.xyz-old_velocity-params.gravity.xyz*dt);
        air_moment+=angular_momentum(p,mass)-old_momentum;
        let lateral=cross(direction,p.normal.xyz);
        let count=select(4u,7u,p.air_torque.w>0.0);
        let uv=array<vec2<f32>,7>(vec2<f32>(-0.5,0.0),vec2<f32>(-1.0/6.0,0.5),vec2<f32>(1.0/6.0,0.5),
            vec2<f32>(0.5,0.0),vec2<f32>(1.0/6.0,-0.5),vec2<f32>(-1.0/6.0,-0.5),vec2<f32>(0.0));
        var corners:array<vec3<f32>,7>;
        var lowest=1e30;
        for (var i=0u;i<count;i++) {
            if p.air_torque.w>0.0 {
                let q=uv[i];
                corners[i]=direction*(q.x*p.direction.w)+lateral*(q.y*p.normal.w)
                    +p.normal.xyz*(p.air_torque.w*(4.0*dot(q,q)-0.5));
            } else {
                corners[i]=direction*(p.direction.w*select(-0.5,0.5,(i&1u)!=0u))+lateral*(p.normal.w*select(-0.5,0.5,(i&2u)!=0u));
            }
            lowest=min(lowest,corners[i].y);
        }
        let depth=params.settings.y+a.surface.y-p.position.y-lowest;
        if depth> -0.0005 {
            p.position.y+=max(depth,0.0);
            let up=vec3<f32>(0.0,1.0,0.0);
            var normal_impulses:array<f32,7>;
            var friction_impulses:array<vec3<f32>,7>;
            var bounce:array<f32,7>;
            for (var i=0u;i<count;i++) {
                let v=(p.velocity.xyz+cross(p.angular.xyz,corners[i])).y;
                bounce[i]=select(0.0,-0.02*v,v< -0.5);
            }
            for (var iteration=0u;iteration<4u;iteration++) { for (var i=0u;i<count;i++) {
                let r=corners[i];
                if p.position.y+r.y>params.settings.y+a.surface.y+0.0005 {continue;}
                let contact=p.velocity.xyz+cross(p.angular.xyz,r);
                let inverse=1.0/mass+dot(up,cross(inverse_inertia(p,mass,cross(r,up)),r));
                let next=max(0.0,normal_impulses[i]+(bounce[i]-contact.y)/inverse);
                let j=up*(next-normal_impulses[i]); normal_impulses[i]=next;
                p.velocity=vec4<f32>(p.velocity.xyz+j/mass,p.velocity.w);
                p.angular=vec4<f32>(p.angular.xyz+inverse_inertia(p,mass,cross(r,j)),p.angular.w);
                let slip=p.velocity.xyz+cross(p.angular.xyz,r);
                let horizontal=vec3<f32>(slip.x,0.0,slip.z);
                if length(horizontal)>1e-12 {
                    let tangent=normalize(horizontal);
                    let inverse_t=1.0/mass+dot(tangent,cross(inverse_inertia(p,mass,cross(r,tangent)),r));
                    var next_friction=friction_impulses[i]-tangent*(dot(slip,tangent)/inverse_t);
                    let limit=0.3*normal_impulses[i];
                    if length(next_friction)>limit {next_friction*=limit/length(next_friction);}
                    let friction=next_friction-friction_impulses[i]; friction_impulses[i]=next_friction;
                    p.velocity=vec4<f32>(p.velocity.xyz+friction/mass,p.velocity.w);
                    p.angular=vec4<f32>(p.angular.xyz+inverse_inertia(p,mass,cross(r,friction)),p.angular.w);
                }
            } }
        }
    }
    p.air_force=vec4<f32>(air_impulse/params.settings.x,0.0);
    p.air_torque=vec4<f32>(air_moment/params.settings.x,p.air_torque.w);
    return p;
}

fn element_force(u: vec3<f32>, n: vec3<f32>, rho_area: f32) -> vec3<f32> {
    let speed=length(u); if speed<1e-6 { return vec3<f32>(0.0); }
    let un=dot(u,n);
    let circulation=-1.2*rho_area*un/speed*cross(u,n);
    let drag=0.5*rho_area*(0.4+2.0*(un/speed)*(un/speed))*speed;
    return cross(circulation,u)-u*drag;
}
fn pressure_shift(u: vec3<f32>, d: vec3<f32>, s: vec3<f32>, n: vec3<f32>, blade: f32, span: f32) -> vec3<f32> {
    let tangential=u-n*dot(u,n); let t=length(tangential); let speed=length(u);
    if t<1e-6 || speed<1e-6 { return vec3<f32>(0.0); }
    let e=tangential/t;
    let chord=inverseSqrt((dot(e,d)/blade)*(dot(e,d)/blade)+(dot(e,s)/span)*(dot(e,s)/span));
    return e*(0.25*chord*t/speed);
}
fn rotate(v: vec3<f32>, axis: vec3<f32>, angle: f32) -> vec3<f32> {
    return v*cos(angle)+cross(axis,v)*sin(angle)+axis*(dot(axis,v)*(1.0-cos(angle)));
}
// Six-degree-of-freedom blade held by an elastic petiole: translational joint spring to
// the anchor, bending and torsion from the twig-carried rest frame, four sheltered plate elements.
fn joint_spring(a: Coeff, anchor: Anchor, position: vec3<f32>, velocity: vec3<f32>, omega: vec3<f32>, d: vec3<f32>, arm: f32, time: f32) -> vec3<f32> {
    let joint=-d*arm;
    let pivot=anchor.position.xyz+anchor.velocity.xyz*(time-params.settings.x);
    return (pivot-position-joint)*a.spring.y+(anchor.velocity.xyz-velocity-cross(omega,joint))*a.spring.z;
}
fn hinge(initial: State, a: Coeff, anchor: Anchor, rest: BladeFrame, wind: vec3<f32>) -> State {
    let p=initial;
    let m=a.spring.x; let blade=p.direction.w; let span=p.normal.w; let area=p.angular.w;
    let arm=a.petiole.x+0.5*blade;
    let added=3.141592653589793*1.225*area*blade/4.0;
    let jd=m*span*span/16.0+added*span*span/32.0;
    let js=m*blade*blade/16.0+added*blade*blade/32.0;
    let jn=m*(blade*blade+span*span)/16.0;
    let bend_damping=2.0*a.petiole.w*sqrt(a.petiole.y*(js+(m+added)*arm*arm));
    let twist_damping=2.0*a.petiole.w*sqrt(a.petiole.z*jd);
    let d0=rest.direction.xyz; let n0=rest.normal.xyz; let s0=cross(d0,n0);
    let rho_area=1.225*area*a.surface.z/4.0;
    let dt=params.settings.x; let gravity=params.gravity.xyz;
    let preload=cross(d0*arm,gravity*m);
    var position=p.position.xyz; var velocity=p.velocity.xyz;
    var omega=p.angular.xyz; var d=p.direction.xyz; var n=p.normal.xyz;
    let rate=max(480.0,60.0*length(wind-velocity)+8.0*length(omega));
    let steps=u32(max(1.0,ceil(dt*rate)));
    let h=dt/f32(steps);
    let elements=array<vec2<f32>,4>(vec2<f32>(-0.25,-0.25),vec2<f32>(-0.25,0.25),vec2<f32>(0.25,-0.25),vec2<f32>(0.25,0.25));
    var impulse=vec3<f32>(0.0);
    for (var step=0u;step<steps;step++) {
        let s=cross(d,n);
        var force=gravity*m; var torque=vec3<f32>(0.0);
        for (var e=0u;e<4u;e++) {
            let q=elements[e];
            let r=d*(q.x*blade)+s*(q.y*span);
            let u=velocity+cross(omega,r)-wind;
            let f=element_force(u,n,rho_area);
            impulse+=f*h; force+=f;
            torque+=cross(r+pressure_shift(u,d,s,n,blade,span)+n*(0.5*p.air_torque.w),f);
        }
        let held=joint_spring(a,anchor,position,velocity,omega,d,arm,h*f32(step));
        force+=held;
        let sine=0.5*(cross(d0,d)+cross(s0,s)+cross(n0,n));
        let cosine=0.5*(dot(d0,d)+dot(s0,s)+dot(n0,n)-1.0);
        let magnitude=length(sine);
        var theta=sine; if magnitude>1e-7 { theta=sine*(atan2(magnitude,cosine)/magnitude); }
        var axis=d; if length(d+d0)>1e-6 { axis=normalize(d+d0); }
        let twist=axis*dot(theta,axis);
        let relative=omega-rest.angular.xyz;
        let relative_twist=axis*dot(relative,axis);
        torque+=cross(-d*arm,held)-preload
            -(theta-twist)*a.petiole.y-twist*a.petiole.z
            -(relative-relative_twist)*bend_damping-relative_twist*twist_damping;
        velocity+=((force-n*dot(force,n))/m+n*(dot(force,n)/(m+added)))*h;
        position+=velocity*h;
        let momentum=d*(jd*dot(omega,d))+s*(js*dot(omega,s))+n*(jn*dot(omega,n));
        let net=torque-cross(omega,momentum);
        omega+=(d*(dot(net,d)/jd)+s*(dot(net,s)/js)+n*(dot(net,n)/jn))*h;
        let spin=length(omega); if spin>400.0 { omega*=400.0/spin; }
        let angle=length(omega)*h;
        if angle>1e-9 { let k=normalize(omega); d=rotate(d,k,angle); n=rotate(n,k,angle); }
        d=normalize(d); n=normalize(n-d*dot(n,d));
    }
    let air=impulse/dt;
    let joint=joint_spring(a,anchor,position,velocity,omega,d,arm,dt);
    let load=length(joint);
    let attached=load<=a.spring.w*params.settings.z;
    let age=p.velocity.w+select(dt,0.0,attached);
    return State(vec4<f32>(position,select(0.0,1.0,attached)),vec4<f32>(velocity,age),
        vec4<f32>(select(vec3<f32>(0.0),joint,attached),max(p.force.w,load)),vec4<f32>(air,0.0),
        vec4<f32>(0.0,0.0,0.0,p.air_torque.w),vec4<f32>(d,blade),vec4<f32>(n,span),vec4<f32>(omega,area));
}

fn step_particle(i: u32, initial: State, a: Coeff, anchor: Anchor, wind: vec3<f32>) -> State {
    var p=initial;
    if p.position.w>0.5 && anchor.position.w>0.5 && p.direction.w>0.0 && params.wind.w>0.5 && a.petiole.y>0.0 {
        return hinge(p,a,anchor,blade_frames[i],wind);
    }
    if p.position.w>0.5 && p.direction.w>0.0 && params.wind.w>0.5 {
        p.direction=vec4<f32>(blade_frames[i].direction.xyz,p.direction.w);
        p.normal=vec4<f32>(blade_frames[i].normal.xyz,p.normal.w);
        p.angular=vec4<f32>(blade_frames[i].angular.xyz,p.angular.w);
    }
    let mass=a.spring.x; let k=a.spring.y; let damping=a.spring.z;
    let dt=params.settings.x;
    let drag=a.surface.x*length(p.velocity.xyz-wind);
    let momentum=mass*p.velocity.xyz+dt*(mass*params.gravity.xyz+drag*wind);
    let free=momentum/(mass+dt*drag);
    var v=free; var force=vec3<f32>(0.0); var peak=p.force.w;
    var attached=p.position.w>0.5 && anchor.position.w>0.5;
    var age=p.velocity.w;
    if attached {
        v=(momentum+dt*(k*(anchor.position.xyz-p.position.xyz)+damping*anchor.velocity.xyz))
            /(mass+dt*(drag+damping)+dt*dt*k);
        force=k*(anchor.position.xyz-p.position.xyz-dt*v)+damping*(anchor.velocity.xyz-v);
        let load=length(force); peak=max(peak,load);
        attached=load<=a.spring.w*params.settings.z;
    }
    if !attached { v=free; force=vec3<f32>(0.0); age+=dt; }
    if !attached && p.direction.w>0.0 {
        p.position.w=0.0; p.velocity.w=age; p.force=vec4<f32>(0.0,0.0,0.0,peak);
        return fly_leaf(p,a,wind);
    }
    let air_force=drag*(wind-v);
    var x=p.position.xyz+dt*v;
    let floor=params.settings.y+a.surface.y;
    if x.y<floor {
        x.y=floor;
        if v.y<0.0 { v.y*=-0.08; }
        let friction=exp(-18.0*dt); v.x*=friction; v.z*=friction;
    }
    return State(vec4<f32>(x,select(0.0,1.0,attached)),vec4<f32>(v,age),vec4<f32>(force,peak),vec4<f32>(air_force,0.0),vec4<f32>(0.0,0.0,0.0,p.air_torque.w),p.direction,p.normal,p.angular);
}

// Air at the particle: the coupled MAC field, the uploaded per-particle wind or uniform wind.
fn local_air(i: u32, point: vec3<f32>) -> vec3<f32> {
    if params.coupling.x>0.5 { return sample_exterior(point)+turbulence(point); }
    return select(params.wind.xyz,local_wind[i].xyz,params.settings.w>0.5);
}
fn quantize(value: f32) -> i32 { return i32(round(value/REACTION_QUANTUM)); }
fn record_boundary(lane: u32, impulse: vec3<f32>) {
    for (var k=0u;k<3u;k++) { atomicAdd(&boundary[lane+k],quantize(impulse[k])); }
}
// Trilinear MAC-face deposit of a force, matching `AirForces::add`.
fn deposit(point: vec3<f32>, force: vec3<f32>) {
    let h=cfg.origin_h.w;
    for (var axis=0u;axis<3u;axis++) {
        var limit=vec3<f32>(cfg.cells.xyz)-vec3<f32>(1.0); limit[axis]+=1.0;
        let s=clamp((point-cfg.origin_h.xyz)/h-offset(axis),vec3<f32>(0.0),limit);
        let base=min(floor(s),limit-vec3<f32>(1.0)); let f=s-base;
        for (var corner=0u;corner<8u;corner++) {
            var q=base; var weight=1.0;
            for (var k=0u;k<3u;k++) {
                if ((corner>>k)&1u)==1u { q[k]+=1.0; weight*=f[k]; } else { weight*=1.0-f[k]; }
            }
            atomicAdd(&reactions[3u*index(vec3<u32>(q))+axis],quantize(weight*force[axis]));
        }
    }
}
fn react(point: vec3<f32>, force: vec3<f32>) {
    let h=cfg.origin_h.w;
    let inner=cfg.origin_h.xyz+vec3<f32>(h);
    let outer=cfg.origin_h.xyz+vec3<f32>(cfg.cells.xyz)*h-vec3<f32>(h);
    if all(point>=inner) && all(point<=outer) { deposit(point,force*params.coupling.z); }
    else { record_boundary(0u,force*params.settings.x); }
}
// Pure couple as symmetric force pairs on three axes, matching `AirForces::add_torque`.
fn react_torque(point: vec3<f32>, torque: vec3<f32>) {
    if dot(torque,torque)<1e-24 { return; }
    let h=cfg.origin_h.w; let margin=vec3<f32>(2.0*h);
    let outer=cfg.origin_h.xyz+vec3<f32>(cfg.cells.xyz)*h-margin;
    if all(point>=cfg.origin_h.xyz+margin) && all(point<=outer) {
        let axes=array<vec3<f32>,3>(vec3<f32>(1.0,0.0,0.0),vec3<f32>(0.0,1.0,0.0),vec3<f32>(0.0,0.0,1.0));
        for (var k=0u;k<3u;k++) {
            let force=cross(torque,axes[k])/(4.0*h)*params.coupling.z;
            deposit(point+axes[k]*h,force);
            deposit(point-axes[k]*h,-force);
        }
    } else { record_boundary(4u,torque*params.settings.x); }
}

@compute @workgroup_size(64)
fn advance(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x; if i>=arrayLength(&states) { return; }
    let p=states[i];
    let wind=local_air(i,p.position.xyz);
    var next=step_particle(i,p,coefficients[i],anchors[i],wind);
    next.air_force.w=length(wind);
    if params.coupling.y>0.5 {
        react(p.position.xyz,-next.air_force.xyz);
        react_torque(next.position.xyz,-next.air_torque.xyz);
    }
    states[i]=next;
}
