/// Assign escape links to the ordered children of a validated, bounded-depth tree.
pub(super) fn thread_tree(
    root: Option<u32>,
    children: impl Fn(u32) -> Option<[u32; 2]>,
    mut write_escape: impl FnMut(u32, u32),
) {
    fn visit(
        node: u32,
        escape: u32,
        children: &impl Fn(u32) -> Option<[u32; 2]>,
        write_escape: &mut impl FnMut(u32, u32),
    ) {
        write_escape(node, escape);
        if let Some([left, right]) = children(node) {
            visit(left, right, children, write_escape);
            visit(right, escape, children, write_escape);
        }
    }
    if let Some(root) = root {
        visit(root, u32::MAX, &children, &mut write_escape);
    }
}

#[cfg(test)]
mod tests {
    use super::thread_tree;

    #[test]
    fn empty_and_single_leaf_links_terminate() {
        thread_tree(None, |_| unreachable!(), |_, _| unreachable!());
        let mut links = [0];
        thread_tree(Some(0), |_| None, |node, next| links[node as usize] = next);
        assert_eq!(links, [u32::MAX]);
    }

    #[test]
    fn escape_walk_matches_stack_walk_for_every_pruning_mask() {
        for (root, children) in [
            (
                6u32,
                [
                    None,
                    None,
                    None,
                    None,
                    Some([0, 1]),
                    Some([2, 3]),
                    Some([4, 5]),
                ],
            ),
            (
                0u32,
                [
                    Some([1, 2]),
                    Some([3, 4]),
                    None,
                    Some([5, 6]),
                    None,
                    None,
                    None,
                ],
            ),
        ] {
            let mut links = [u32::MAX; 7];
            thread_tree(
                Some(root),
                |i| children[i as usize],
                |i, next| links[i as usize] = next,
            );
            for mask in 0..128 {
                let accepted = |i: u32| mask & (1 << i) != 0;
                let mut stack = vec![root];
                let mut expected = Vec::new();
                while let Some(i) = stack.pop() {
                    expected.push(i);
                    if accepted(i)
                        && let Some([left, right]) = children[i as usize]
                    {
                        stack.extend([right, left]);
                    }
                }
                let mut actual = Vec::new();
                let mut i = root;
                while i != u32::MAX {
                    assert!(actual.len() < children.len());
                    actual.push(i);
                    i = if accepted(i) {
                        children[i as usize].map_or(links[i as usize], |pair| pair[0])
                    } else {
                        links[i as usize]
                    };
                }
                assert_eq!(actual, expected, "root {root}, mask {mask}");
            }
        }
    }
}
