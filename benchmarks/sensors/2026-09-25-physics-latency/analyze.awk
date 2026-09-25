BEGIN {
    FS = OFS = "\t"
    OFMT = CONVFMT = "%.12g"
    if (repeats == "") repeats = 3
    if (frames == "") frames = 64
    if (warmups == "") warmups = 8
    if (case_count == "") case_count = 48
    group_frames = 1 + warmups + frames
    if (timed !~ /^[01]$/)
        fail("set timed=0 or timed=1")
    print "case", "repeat", "n", "complete_p50_us", "complete_p95_us", "physics_cpu_p50_us", "sensor_encode_p50_us", "queue_span_p50_us", "allocations_p50", "reallocations_p50", "requested_bytes_p50"
}

function fail(message) {
    print FILENAME ":" FNR ": " message > "/dev/stderr"
    invalid = 1
    exit 1
}

function quantile(values, p, sorted, n) {
    n = asort(values, sorted, "@val_num_asc")
    return sorted[int((n * p + 99) / 100)]
}

FNR == 1 {
    if (NF != 25 || $1 != "case" || $11 != "complete_us" || $25 != "requested_bytes")
        fail("unexpected header")
    next
}

{
    if (NF != 25 || !match($1, /^(cpu|gpu)_a(1|32)_steps(1|10)_(lidar1k|cloud100k_camera640|cloud1m_camera4x720_aperture4)_(device|host)$/, c))
        fail("invalid case or column count")
    if (case_filter != "" && !index($1, case_filter))
        fail("unexpected filtered case")
    for (column = 2; column <= 25; column++) {
        if (column == 3 || (column == 12 && !timed && $column == "NA"))
            continue
        if ($column !~ /^[0-9]+([.][0-9]+)?$/)
            fail("nonfinite or malformed numeric column " column)
    }
    if ($2 !~ /^[0-9]+$/ || $2 >= repeats || $4 !~ /^[0-9]+$/ || $4 >= group_frames)
        fail("invalid repeat or frame index")
    if (!timed && $12 != "NA")
        fail("unexpected timestamp in control run")
    phase = $4 == 0 ? "first" : ($4 <= warmups ? "warmup" : "sample")
    if ($3 != phase || seen[$1, $2, $4]++)
        fail("invalid phase or duplicate frame")
    gpu = c[1] == "gpu"
    host = c[5] == "host"
    small = c[4] == "lidar1k"
    medium = c[4] == "cloud100k_camera640"
    mounts = small ? 1 : (medium ? 4 : 16)
    cameras = small ? 0 : (medium ? 1 : 4)
    rays = small ? 1024 : (medium ? 100096 : 1000448)
    pixels = small ? 0 : (medium ? 307200 : 3686400)
    payload = small ? 52 * rays : 116 * rays + 4 * (1 + 5 * mounts + 6) + 64 * 4096 + 32 * pixels
    if ($5 != rays || $6 != pixels || $7 != 4 * c[2])
        fail("workload dimension mismatch")
    if ($13 != (gpu ? 0 : 32 * $7) || $14 != 128 * mounts + 32 * cameras || $15 != (gpu ? 48 * c[3] : 0))
        fail("upload accounting mismatch")
    if ($16 != (host ? payload : 0) || $17 != (gpu ? 5 * c[3] : 0) || $18 != 1 || $19 != host || $20 != timed)
        fail("transfer/submission accounting mismatch")
    if ($21 !~ /^[01]$/ || (!host && $21 != 1) || $22 <= 0)
        fail("wait/dispatch accounting mismatch")
    if ($11 < $10 || $10 < $8 || $11 <= 0 || (timed && $12 <= 0))
        fail("invalid timing boundaries")
    group = $1 SUBSEP $2
    counts[group]++
    cases[$1] = 1
    phases[$3]++
    if ($3 == "sample") {
        sample_index = ++samples[group]
        complete[group][sample_index] = $11 + 0
        physics[group][sample_index] = $8 + 0
        encode[group][sample_index] = $9 + 0
        if (timed)
            spans[group][sample_index] = $12 + 0
        allocations[group][sample_index] = $23 + 0
        reallocations[group][sample_index] = $24 + 0
        bytes[group][sample_index] = $25 + 0
    }
}

END {
    if (invalid)
        exit 1
    groups = case_count * repeats
    if (length(cases) != case_count || length(counts) != groups || phases["first"] != groups || phases["warmup"] != groups * warmups || phases["sample"] != groups * frames)
        fail("incomplete matrix")
    PROCINFO["sorted_in"] = "@ind_str_asc"
    for (group in counts) {
        if (counts[group] != group_frames || samples[group] != frames)
            fail("incomplete case/repeat")
        split(group, parts, SUBSEP)
        p95 = quantile(complete[group], 95)
        name = parts[1]
        if (!(name in minimum) || p95 < minimum[name])
            minimum[name] = p95
        if (p95 > maximum[name])
            maximum[name] = p95
        print name, parts[2], samples[group], quantile(complete[group], 50), p95, quantile(physics[group], 50), quantile(encode[group], 50), timed ? quantile(spans[group], 50) : "NA", quantile(allocations[group], 50), quantile(reallocations[group], 50), quantile(bytes[group], 50)
    }
    for (name in cases) {
        spread = (maximum[name] / minimum[name] - 1) * 100
        noisy += spread > 10
        print "spread", name, spread > "/dev/stderr"
    }
    print "validated cases=" case_count " groups=" groups " first=" phases["first"] " warmup=" phases["warmup"] " samples=" phases["sample"] " p95_spread_over_10_percent=" noisy + 0 > "/dev/stderr"
}
