def dot($a; $b): [range(0; 3) | $a[.] * $b[.]] | add;
def tilt($begin; $end; $direction):
  [.samples[] | select(.time >= $begin and .time < $end) |
    dot(.thrust_axis; $direction)];

. as $report |
{
  full_movie: (.seconds == 30 and .frames == 900 and .width == 1920 and .height == 1080),
  motion_verified: .verified,
  finite: ([.samples[] | [.position[], .velocity[], .orientation_xyzw[], .air_force[], .rpm[]][] |
    (isfinite and (isnan | not))] | all),
  lateral_acceleration: ((tilt(3; 4.2; [1,0,0]) | max) > 0.05),
  lateral_braking: ((tilt(5.8; 7; [1,0,0]) | min) < -0.05),
  forward_acceleration: ((tilt(9; 10.2; [0,0,1]) | max) > 0.05),
  forward_braking: ((tilt(11.8; 13; [0,0,1]) | min) < -0.05),
  diagonal_acceleration: ((tilt(15; 16.5; [-0.8,0,-0.6]) | max) > 0.05),
  diagonal_braking: ((tilt(18.5; 20; [-0.8,0,-0.6]) | min) < -0.05),
  opposite_air_reaction: ([.samples[] | dot(.air_force; .thrust_axis) < 0] | all),
  rotor_force_balance: (.peak_rotor_air_force_balance_error_n < 1e-9),
  stable_finish: (.final_error_m < 0.08 and .final_speed_m_s < 0.08)
} as $checks |
{checks: $checks, passed: ($checks | all),
 peak_tracking_error_m: $report.peak_tracking_error_m,
 peak_tilt_degrees: $report.peak_tilt_degrees,
 final_error_m: $report.final_error_m, final_speed_m_s: $report.final_speed_m_s} |
if .passed then . else error("flight motion acceptance failed: \(.)") end
