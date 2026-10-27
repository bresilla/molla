pub fn is_smoke(args: &[String]) -> bool {
    !args.iter().any(|arg| arg == "--bench")
        || args
            .iter()
            .any(|arg| matches!(arg.as_str(), "--smoke" | "--test"))
}

pub fn defaults(args: &[String]) -> (usize, usize, usize) {
    if is_smoke(args) {
        (2, 1, 1)
    } else {
        (64, 8, 3)
    }
}
