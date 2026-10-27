        var candidate=coop_natural_merit();
        if rate_coefficient!=0.0 {
            for (var slot=coop_lane;slot<shared_status.control.y;slot+=64u) {
                let c=shared_history[slot].active_index;
                let p=shared_history[c].baseline_k; let w=shared_history[c].baseline_w;
                let trial_p=p.xyz+rate_coefficient*(p.xyz-shared_history[c].previous_k.xyz);
                let trial_w=w.xyz+rate_coefficient*(w.xyz-shared_history[c].previous_w.xyz);
                let lambda=dot(-trial_p,normalize(shared_contacts[c].normal_gap.xyz));
                shared_set_impulses(c,vec4<f32>(trial_p,lambda),vec4<f32>(trial_w,0.0));
            }
            storageBarrier();
            coop_rebuild_rates();
            let rate_merit=coop_natural_merit();
            if rate_merit<candidate { candidate=rate_merit; }
            else {
                let second=bitcast<f32>(shared_status.control.w);
                for (var slot=coop_lane;slot<shared_status.control.y;slot+=64u) {
                    let c=shared_history[slot].active_index;
                    let p=shared_history[c].baseline_k; let w=shared_history[c].baseline_w;
                    let trial_p=p.xyz+coefficient*(p.xyz-shared_history[c].previous_k.xyz)+second*(shared_history[c].previous_k.xyz-shared_history[c].full_older_p.xyz);
                    let trial_w=w.xyz+coefficient*(w.xyz-shared_history[c].previous_w.xyz)+second*(shared_history[c].previous_w.xyz-shared_history[c].full_older_w.xyz);
                    let lambda=dot(-trial_p,normalize(shared_contacts[c].normal_gap.xyz));
                    shared_set_impulses(c,vec4<f32>(trial_p,lambda),vec4<f32>(trial_w,0.0));
                }
                storageBarrier();
                coop_rebuild_rates();
            }
        }
