#!/usr/bin/env bash
set -euo pipefail
video=${1:?Provide the detailed showcase MP4}
manifest=${video%.*}.json
directory=$(dirname "$video")
for span in 21 11 37 55; do
    start=$(jq -r --argjson span "$span" '[.events[] | select(.kind == "bend_started" and .data.span == $span)][0].video_seconds' "$manifest")
    ffmpeg -v error -y -ss "$start" -t 6 -i "$video" -vf 'fps=2,scale=480:-1,tile=4x3:padding=4:margin=4:color=0x0f151f' -frames:v 1 "$directory/bending-$span-sequence.png"
done
for span in 32 41 56; do
    start=$(jq -r --argjson span "$span" '[.events[] | select(.kind == "rough_break" and .data.span == $span)][0].video_seconds' "$manifest")
    ffmpeg -v error -y -ss "$start" -t 1.6 -i "$video" -vf 'fps=10,scale=640:-1,tile=4x4:padding=4:margin=4:color=0x0f151f' -frames:v 1 "$directory/fracture-$span-sequence.png"
done
start=$(jq -r '[.events[] | select(.kind == "crown_trunk_cut")][0].video_seconds - 0.25' "$manifest")
ffmpeg -v error -y -ss "$start" -t 8 -i "$video" -vf 'fps=4,scale=480:-1,tile=8x4:padding=4:margin=4:color=0x0f151f' -frames:v 1 "$directory/crown-fall-sequence.png"
