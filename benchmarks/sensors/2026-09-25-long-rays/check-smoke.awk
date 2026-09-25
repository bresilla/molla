BEGIN { FS = "\t" }
function fail(message) {
    print FILENAME ":" FNR ": " message > "/dev/stderr"
    invalid = 1
    exit 1
}
FNR == 1 {
    if ($1 != "case" || $2 != "repeat" || $3 != "phase" || $4 != "index")
        fail("unexpected header")
    columns = NF
    next
}
{
    if (NF != columns || $2 != 0 || ($3 != "first" && $3 != "sample"))
        fail("invalid columns, repeat or phase")
    if ($4 != ($3 == "first" ? 0 : 1) || seen[$1, $3]++)
        fail("invalid index or duplicate case/phase")
    for (i = 2; i <= NF; i++) {
        if (i == 3 || (i == timestamp_column && $i == "NA")) continue
        if ($i !~ /^[0-9]+([.][0-9]+)?$/) fail("invalid numeric field " i)
    }
    cases[$1] = 1
    phases[$3]++
    rows++
}
END {
    if (invalid) exit 1
    if (length(cases) != expected || rows != 2 * expected || phases["first"] != expected || phases["sample"] != expected)
        fail("incomplete matrix")
    print FILENAME, "cases=" length(cases), "rows=" rows, "first=" phases["first"], "samples=" phases["sample"], "violations=0"
}
