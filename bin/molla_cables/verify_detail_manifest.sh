#!/usr/bin/env bash
set -euo pipefail
manifest=${1:?Provide the detailed showcase manifest}
jq -e '
    def events($kind): [.events[] | select(.kind == $kind)];
    .sequence == "bending-closeups-large-cuts-v2" and
    .rough_breaks == 4 and .clean_cuts == 6 and .trunk_components == 3 and
    .leaves_total == 420 and .leaf_strength_multiplier_vs_original == 2.2 and
    .opening_leaves_shed >= 5 and .opening_leaves_shed <= 60 and
    ((.mass_before - .mass_after) | fabs) < 1e-8 and .floor_clearance_m >= -0.01 and
    (events("bend_started") | length) == 4 and
    (events("bend_started") | map(.data.span) | unique | length) == 4 and
    (events("bend_started") | map(.data.direction) | unique | length) == 4 and
    (events("bend_released") | length) == 4 and
    (events("bend_released") | all(.data.peak_excursion_m >= 0.025 and .data.new_stress_failures == 0)) and
    (events("bend_recovery") | length) == 4 and
    (events("bend_recovery") | all(.data.free_motion_frames >= 60)) and
    (events("fracture_closeup") | length) == 4 and
    (events("fracture_closeup") | all(.data.distance_m <= 0.55)) and
    (events("rough_break") | length) == 4 and
    (events("rough_break")[0].video_seconds - events("grab_released")[0].video_seconds) >= 0.99 and
    (events("released_control")[0].data | .failures == 0 and .control_seconds >= 3) and
    (events("delayed_load_failure") | length) == 1 and
    events("delayed_load_failure")[0].frame > events("rough_break")[0].frame and
    (events("large_branch_half_cut") | length) == 2 and
    (events("remaining_branch_base_cut") | length) == 2 and
    (events("crown_trunk_cut") | length) == 1 and
    (events("lower_trunk_cut") | length) == 1 and
    (events("fall_observed") | length) == 6 and
    (events("fall_observed") | all(.data.settle_frames >= 210)) and
    events("crown_trunk_cut")[0].frame < events("large_branch_half_cut")[-1].frame and
    events("crown_trunk_cut")[0].data.detached_cells >= 10 and
    events("rough_break")[-1].frame < events("large_branch_half_cut")[0].frame
' "$manifest" >/dev/null
printf 'PASS choreography: multiple bends, close-up fractures, six large cuts, early crown fall\n'
