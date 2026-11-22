def norm: map(. * .) | add | sqrt;
. as $r | .spray as $s |
{
  complete: ($s.phase == "Complete"),
  all_sixteen_stops: (($s.dose_seconds_per_tree|length)==16 and all($s.dose_seconds_per_tree[]; .>=1.999 and .<=2.02)),
  hovering_when_spraying: ($s.max_spray_speed_m_s < 0.1 and $s.max_spray_position_error_m < 0.1),
  stopped_at_finish: (($r.drone_velocity|norm)<0.1),
  dose_total: (($s.emitted_litres-32*($s.flow_l_s // 0.012))|fabs)<0.002,
  reservoir_conserved: (($s.tank_litres+$s.emitted_litres-0.75)|fabs)<1e-8,
  water_conserved: (($s.airborne_litres+$s.canopy_litres+$s.ground_litres-$s.emitted_litres)|fabs)<1e-8,
  canopy_contact: ($s.canopy_litres>0 and $s.wet_leaves>10),
  each_tree_received_water: (($s.deposition_per_tree_litres|length)==16 and all($s.deposition_per_tree_litres[]; .>0)),
  nonnegative_liquid: (all([$s.tank_litres,$s.emitted_litres,$s.canopy_litres,$s.ground_litres,$s.airborne_litres][]; isfinite and .>=0)),
  no_particle_cap: ($s.capacity_stops==0 and $s.peak_parcels<($s.parcel_capacity // 16384)),
  finite: $s.finite,
  trees_intact: ($r.root_error_m==0 and $r.wood_failures==0),
  leaf_close_up: ($r.focus_leaf == null or ($r.focus_leaf as $f | $f.tree==1 and $f.shown_parcels>0
    and $f.final_parcels>=$f.shown_parcels and $f.shown_equivalent_droplets>0
    and ($f.window_s[1]-$f.window_s[0])>10)),
  vapor_hidden: ($r.airflow_visible==false)
} as $checks | {checks:$checks,passed:all($checks[];.),spray:$s} |
if .passed then . else error("AG spray verification failed: \(.)") end
