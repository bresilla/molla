#!/usr/bin/env bash
set -euo pipefail
video=${1:?Provide the showcase MP4}
manifest=${video%.*}.json
probe=${video%.*}.ffprobe.json
directory=$(dirname "$video")
if jq -e '.sequence == "bending-closeups-large-cuts-v2"' "$manifest" >/dev/null; then
    bash "$(dirname "$0")/verify_detail_manifest.sh" "$manifest"
else
jq -e '
    .rough_breaks == 4 and .trunk_components == 3 and
    .standing_branches_removed and
    ((.mass_before - .mass_after) | fabs) < 1e-8 and
    .floor_clearance_m >= -0.01 and
    ([.events[] | select(.kind == "rough_break")] | length) == 4 and
    ([.events[] | select(.kind == "trunk_cut")] | length) == 2 and
    ([.events[] | select(.kind == "rough_break")][0].video_seconds -
     [.events[] | select(.kind == "grab_released")][0].video_seconds) >= 0.99 and
    ([.events[] | select(.kind == "rough_break")][-1].frame <
     [.events[] | select(.kind == "whole_branch_cut")][0].frame) and
    ([.events[] | select(.kind == "whole_branch_cut")][-1].frame <
     [.events[] | select(.kind == "only_trunk")][0].frame) and
    ([.events[] | select(.kind == "only_trunk")][0].frame <
     [.events[] | select(.kind == "trunk_cut")][0].frame)
' "$manifest" >/dev/null
fi
ffprobe -v error -show_streams -show_format -of json "$video" > "$probe"
jq -e --slurpfile manifest "$manifest" '
    .streams | map(select(.codec_type == "video")) | length == 1
' "$probe" >/dev/null
jq -e --slurpfile manifest "$manifest" '
    .streams[] | select(.codec_type == "video") |
    .codec_name == "h264" and .width == 1920 and .height == 1080 and
    .pix_fmt == "yuv420p" and .r_frame_rate == "30/1" and
    .color_space == "bt709" and
    (.nb_frames | tonumber) == $manifest[0].frames and
    ((.duration | tonumber) - $manifest[0].duration_seconds | fabs) < 0.001
' "$probe" >/dev/null
ffmpeg -v error -xerror -i "$video" -map 0:v:0 -f null -
duration=$(jq -r '.duration_seconds' "$manifest")
release=$(jq -r '[.events[] | select(.kind == "grab_released")][0].video_seconds - 0.1 | if . < 0 then 0 else . end' "$manifest")
cut=$(jq -r '[.events[] | select(.kind == "whole_branch_cut" or .kind == "large_branch_half_cut")][0].video_seconds - 0.5' "$manifest")
trunk=$(jq -r '[.events[] | select(.kind == "trunk_cut" or .kind == "crown_trunk_cut" or .kind == "lower_trunk_cut")][-1].video_seconds - 0.5' "$manifest")
ffmpeg -v error -y -i "$video" -vf "fps=24/$duration,scale=480:-1,tile=4x6:padding=8:margin=8:color=0x0f151f" -frames:v 1 "$directory/contact-sheet.png"
ffmpeg -v error -y -ss "$release" -t 4 -i "$video" -vf 'fps=8,scale=480:-1,tile=8x4:padding=4:margin=4:color=0x0f151f' -frames:v 1 "$directory/delayed-failure-sequence.png"
ffmpeg -v error -y -ss "$cut" -t 3 -i "$video" -vf 'fps=8,scale=480:-1,tile=6x4:padding=4:margin=4:color=0x0f151f' -frames:v 1 "$directory/exact-cut-sequence.png"
ffmpeg -v error -y -ss "$trunk" -t 6 -i "$video" -vf 'fps=4,scale=480:-1,tile=6x4:padding=4:margin=4:color=0x0f151f' -frames:v 1 "$directory/trunk-sequence.png"
printf 'PASS video: H.264/yuv420p, 1920x1080, 30 fps, exact frame count/duration, complete decode\n'
