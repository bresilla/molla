use strict;
use warnings;
use JSON::PP;
use File::Temp qw(tempdir);

sub read_bytes {
    my ($path) = @_;
    open my $file, '<:raw', $path or die "$path: $!\n";
    return do { local $/; <$file> };
}

sub write_bytes {
    my ($path, $bytes) = @_;
    open my $file, '>:raw', $path or die "$path: $!\n";
    print {$file} $bytes;
}

sub audit {
    my ($directory) = @_;
    my @names = qw(body_q body_qd joint_q joint_qd coordinate_error velocity_delta position_error_xyz);
    my @files = glob "$directory/step-*-variant-*.*";
    die "expected exactly sixteen binary/layout pairs\n" unless @files == 32;
    my ($schema, $total, @steps) = ('', 0);
    my %totals = map { $_ => { different_words => 0, max_absolute_error => 0 } } @names;
    for my $step (0 .. 7) {
        my @data;
        for my $variant (0 .. 1) {
            my $prefix = "$directory/step-$step-variant-$variant";
            my $layout = read_bytes("$prefix.tsv");
            $schema = $layout unless length $schema;
            die "layout drift at $prefix\n" unless $layout eq $schema;
            my @lines = split /\n/, $layout;
            die "field count\n" unless @lines == @names;
            my $offset = 0;
            for my $index (0 .. $#lines) {
                my @columns = split /\t/, $lines[$index], -1;
                die "invalid field columns\n" unless @columns == 3;
                my ($start, $count, $name) = @columns;
                die "invalid field\n" unless defined($name) && $name eq $names[$index]
                    && $start =~ /^\d+$/ && $count =~ /^[1-9]\d*$/ && $start == $offset;
                $offset += $count;
            }
            my $bytes = read_bytes("$prefix.bin");
            die "binary size does not match fields\n" unless length($bytes) == 4 * $offset;
            my @words = unpack 'V*', $bytes;
            die "non-finite captured state\n" if grep { ($_ & 0x7f800000) == 0x7f800000 } @words;
            push @data, \@words;
        }
        my %fields;
        my $different = 0;
        for my $line (split /\n/, $schema) {
            my ($start, $count, $name) = split /\t/, $line;
            my ($changed, $maximum) = (0, 0);
            for my $index ($start .. $start + $count - 1) {
                my ($left, $right) = ($data[0][$index], $data[1][$index]);
                next if $left == $right;
                $changed++;
                my $error = abs(unpack('f<', pack('V', $left)) - unpack('f<', pack('V', $right)));
                $maximum = $error if $error > $maximum;
            }
            $fields{$name} = { words => 0 + $count, different_words => $changed, max_absolute_error => $maximum };
            $different += $changed;
            $totals{$name}{different_words} += $changed;
            $totals{$name}{max_absolute_error} = $maximum if $maximum > $totals{$name}{max_absolute_error};
        }
        $total += $different;
        push @steps, { step => $step, different_words => $different, fields => \%fields };
    }
    return { schema => 1, observations => 8, different_words => $total,
        exact_parity => $total == 0 ? JSON::PP::true : JSON::PP::false,
        fields => \%totals, steps => \@steps };
}

sub check_log {
    my ($report, $log) = @_;
    my @counts = $log =~ /mismatched words=(\d+); measured GPU ms/g;
    die "missing or inconsistent comparison record\n" unless @counts == 1 && $counts[0] == $report->{different_words};
}

if (@ARGV == 1 && $ARGV[0] eq '--self-test') {
    my $directory = tempdir(CLEANUP => 1);
    my @names = qw(body_q body_qd joint_q joint_qd coordinate_error velocity_delta position_error_xyz);
    my $layout = join '', map { "$_\t1\t$names[$_]\n" } 0 .. $#names;
    for my $step (0 .. 7) {
        for my $variant (0 .. 1) {
            write_bytes("$directory/step-$step-variant-$variant.tsv", $layout);
            write_bytes("$directory/step-$step-variant-$variant.bin", pack('V*', (0) x 7));
        }
    }
    die "positive fixture\n" unless audit($directory)->{exact_parity};
    my $prefix = "$directory/step-3-variant-1";
    write_bytes("$prefix.bin", pack('V*', 0x3f800000, (0) x 6));
    my $changed = audit($directory);
    die "mismatch fixture\n" unless $changed->{different_words} == 1
        && $changed->{fields}{body_q}{max_absolute_error} == 1 && !$changed->{exact_parity};
    my $valid_log = 'mismatched words=1; measured GPU ms';
    check_log($changed, $valid_log);
    for my $log ('', 'mismatched words=0; measured GPU ms', "$valid_log $valid_log") {
        die "accepted invalid comparison record\n" if eval { check_log($changed, $log); 1 };
    }
    for my $mutation ('layout', 'truncated', 'nonfinite') {
        write_bytes("$prefix.tsv", $layout);
        write_bytes("$prefix.bin", pack('V*', (0) x 7));
        write_bytes("$prefix.tsv", "broken\n") if $mutation eq 'layout';
        write_bytes("$prefix.bin", 'x') if $mutation eq 'truncated';
        write_bytes("$prefix.bin", pack('V*', 0x7fc00000, (0) x 6)) if $mutation eq 'nonfinite';
        die "accepted $mutation fixture\n" if eval { audit($directory); 1 };
    }
    print "capture audit: exact, mismatch, layout, truncation, nonfinite and log-consistency fixtures pass\n";
    exit 0;
}

die "usage: $0 STATE_DIRECTORY [MACHINE_LOG]\n" unless @ARGV >= 1 && @ARGV <= 2;
my $report = audit($ARGV[0]);
if (@ARGV == 2) {
    check_log($report, read_bytes($ARGV[1]));
}
print JSON::PP->new->canonical->pretty->encode($report);
exit($report->{different_words} ? 1 : 0);
