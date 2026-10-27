fn hybrid_direction(id:u32) {
    solve_bounded(id,min(8u,params.iterations));
    if lane==0u {control.count=status.iterations;control.stop=atomicLoad(&status.fault);}
    let used=workgroupUniformLoad(&control.count);
    let fault=workgroupUniformLoad(&control.stop);
    if fault!=8u {return;}
    prepare_condensed();
    if direct_direction() {
        if lane==0u {status.iterations+=used;}
        return;
    }
    if used>=params.iterations {
        if lane==0u {atomicOr(&status.fault,8u);status.iterations=used;}
        return;
    }
    prepare_linear();solve_bounded(id,params.iterations-used);
    if lane==0u {status.iterations+=used;}
}

