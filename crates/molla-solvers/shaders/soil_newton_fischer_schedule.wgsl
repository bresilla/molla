    // Phases: base, Newton trial, damping base, damped trial, final restore.
    var phase=0u;var step=0u;var merit=0.0;var state_norm=0.0;
    var alpha=1.0;var backtrack=0u;var damping=0.0;var attempt=0u;
    loop {
        let candidate=evaluate();
        if phase==4u {if lane==0u {atomicOr(&report.fault,16u);}return;}
        if phase==1u {
            if lane==0u {outer.accepted=select(0u,1u,atomicLoad(&status.fault)==0u && candidate>=0.0 && candidate<merit*(1.0-1e-4*alpha));}
            if workgroupUniformLoad(&outer.accepted)!=0u {
                step++;if lane==0u {report.steps=step;}phase=0u;
                if reduce_fischer_smoothing(candidate) {continue;}
            } else {
                backtrack++;
                if backtrack<16u {
                    alpha*=0.5;
                    if lane==0u {atomicStore(&status.fault,0u);report.trials++;}
                    storageBarrier();trial(alpha);continue;
                }
                damping=merit/(1.0+state_norm);attempt=0u;phase=2u;
                restore();if lane==0u {atomicStore(&status.fault,0u);}storageBarrier();continue;
            }
        }
        if phase==3u {
            if lane==0u {outer.accepted=select(0u,1u,atomicLoad(&status.fault)==0u && candidate>=0.0 && candidate<merit);}
            if workgroupUniformLoad(&outer.accepted)!=0u {
                step++;if lane==0u {report.steps=step;}phase=0u;
                if reduce_fischer_smoothing(candidate) {continue;}
            } else {
                damping*=10.0;attempt++;phase=select(2u,4u,attempt==8u);
                restore();if lane==0u {atomicStore(&status.fault,0u);}storageBarrier();continue;
            }
        }
        if phase==0u {
            merit=candidate;
            if lane==0u {outer.stop=atomicLoad(&status.fault);if outer.stop!=0u || merit<0.0 || !finite(merit) {atomicOr(&report.fault,select(2u,outer.stop,outer.stop!=0u));outer.stop=1u;}}
            if workgroupUniformLoad(&outer.stop)!=0u {return;}
            if certify_candidate(lane) {return;}
            if lane==0u {outer.stop=atomicLoad(&publication_status.publication_fault)&3u;if outer.stop!=0u {atomicOr(&report.fault,outer.stop);}}
            if workgroupUniformLoad(&outer.stop)!=0u {return;}
            if step==params.newton_iterations {if lane==0u {atomicOr(&report.fault,8u);}return;}
            for(var slot=lane;slot<status.active_count;slot+=64u) {
                let c=active_indices[slot];saved[c]=states[c];scratch[c]=Six(states[c].high.k+states[c].low.k,states[c].high.w+states[c].low.w);
            }
            storageBarrier();state_norm=scratch_norm();prepare_linear();solve(lane);
            if lane==0u {report.linear_iterations+=status.iterations;outer.stop=atomicLoad(&status.fault);outer.accepted=0u;}
            if workgroupUniformLoad(&outer.stop)==0u {
                phase=1u;alpha=1.0;backtrack=0u;
                if lane==0u {atomicStore(&status.fault,0u);report.trials++;}
                storageBarrier();trial(alpha);continue;
            }
            damping=merit/(1.0+state_norm);attempt=0u;phase=2u;
            restore();if lane==0u {atomicStore(&status.fault,0u);}storageBarrier();continue;
        }
        prepare_linear();least_squares(lane,sqrt(damping));
        if lane==0u {report.linear_iterations+=status.iterations;report.damping=damping;outer.stop=atomicLoad(&status.fault)&3u;}
        if workgroupUniformLoad(&outer.stop)==0u {
            phase=3u;if lane==0u {atomicStore(&status.fault,0u);report.trials++;}
            storageBarrier();trial(1.0);continue;
        }
        damping*=10.0;attempt++;phase=select(2u,4u,attempt==8u);
        restore();if lane==0u {atomicStore(&status.fault,0u);}storageBarrier();
    }
}
fn reduce_fischer_smoothing(residual:f32)->bool {
    if fischer_smoothing>0.0 && residual<=fischer_smoothing {
        fischer_smoothing*=0.1;
        if fischer_smoothing<1e-10 {fischer_smoothing=0.0;}
        return true;
    }
    return false;
}
