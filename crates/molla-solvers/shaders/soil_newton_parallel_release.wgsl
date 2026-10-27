fn conditioned_parallel_choice(c:u32) {
    var candidate=ReleaseChoice(Six(),3.402823e38,0u);
    if lane<160u {candidate=release_local_candidate(c,lane/20u,lane%20u);}
    parts[lane]=vec2<f32>(select(3.402823e38,candidate.score,candidate.valid!=0u),f32(lane));
    workgroupBarrier();
    for(var stride=128u;stride>0u;stride/=2u) {
        if lane<stride {
            let a=parts[lane];let b=parts[lane+stride];
            if b.x<a.x || (b.x==a.x && b.y<a.y) {parts[lane]=b;}
        }
        workgroupBarrier();
    }
    let threshold=workgroupUniformLoad(&release_best.score);
    if lane==u32(parts[0].y) && candidate.valid!=0u && candidate.score<threshold {
        release_best=candidate;release_contact=c;conditioned_best=select(0u,1u,conditioned_model);
    }
    workgroupBarrier();
}
