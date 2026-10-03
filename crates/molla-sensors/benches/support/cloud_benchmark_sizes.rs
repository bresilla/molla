pub fn is_smoke(args: &[String]) -> bool {
    !args.iter().any(|arg| arg == "--bench") || args.iter().any(|arg| arg == "--test")
}

pub fn sizes(args: &[String]) -> &'static [usize] {
    if is_smoke(args) {
        &[1024, 65_536]
    } else {
        &[1024, 65_536, 1_000_003]
    }
}
