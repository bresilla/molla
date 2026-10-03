pub fn dimensions(width: u32, height: u32, smoke: bool) -> (u32, u32) {
    if smoke {
        ((width / 10).max(1), (height / 10).max(1))
    } else {
        (width, height)
    }
}
