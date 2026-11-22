. as $r | .spray.deposition_per_tree_litres as $d |
{
  wind_announced: ($r.crosswind.announced_s != null and $r.crosswind.full_s != null),
  drift_close_up: ($r.drift_leaf != null and ($r.drift_leaf as $f | $f.tree == 15
    and $f.same_leaf_as_focus and ($f.window_s[1] - $f.window_s[0]) > 10
    and $f.window_s[0] > $r.crosswind.full_s)),
  second_row_lost_most_spray: (($d[8:] | add) < 0.5 * ($d[:8] | add)),
  drift_leaf_received_less: ($r.drift_leaf.shown_equivalent_droplets
    < $r.focus_leaf.shown_equivalent_droplets)
} as $checks | {checks: $checks, passed: all($checks[]; .)} |
if .passed then . else error("crosswind verification failed: \(.)") end
