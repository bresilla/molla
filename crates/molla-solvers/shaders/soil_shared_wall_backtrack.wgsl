        var candidate=coop_natural_merit();
        if candidate>=baseline && baseline<=1e-12 {
            for (var c=coop_lane;c<shared_params.contacts;c+=64u) {
                if shared_contacts[c].soil_mass.w==0.0 { continue; }
                let p=shared_history[c].baseline_k; let w=shared_history[c].baseline_w;
                let trial_p=p.xyz+(0.5*coefficient)*(p.xyz-shared_history[c].previous_k.xyz);
                let trial_w=w.xyz+(0.5*coefficient)*(w.xyz-shared_history[c].previous_w.xyz);
                let lambda=dot(-trial_p,normalize(shared_contacts[c].normal_gap.xyz));
                shared_set_impulses(c,vec4<f32>(trial_p,lambda),vec4<f32>(trial_w,0.0));
            }
            storageBarrier();
            coop_rebuild_rates();
            candidate=coop_natural_merit();
        }
