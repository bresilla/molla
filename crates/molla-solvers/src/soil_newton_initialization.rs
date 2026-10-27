use molla_core::Result;
use std::ops::Range;

#[derive(Debug)]
pub(crate) struct Local {
    pub function: String,
    pub name: String,
    pub declaration: Range<usize>,
}

fn walk(
    block: &naga::Block,
    ranges: &mut Vec<Range<usize>>,
    stores: &mut Vec<(naga::Handle<naga::Expression>, naga::Span)>,
) {
    for (statement, span) in block.span_iter() {
        match statement {
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                if let Some(range) = span.to_range() {
                    ranges.push(range);
                }
                walk(body, ranges, stores);
                walk(continuing, ranges, stores);
            }
            naga::Statement::Block(body) => walk(body, ranges, stores),
            naga::Statement::If { accept, reject, .. } => {
                walk(accept, ranges, stores);
                walk(reject, ranges, stores);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    walk(&case.body, ranges, stores);
                }
            }
            naga::Statement::Store { pointer, .. } => stores.push((*pointer, *span)),
            _ => {}
        }
    }
}

pub(crate) fn audit(source: &str) -> Result<Vec<Local>> {
    let module = naga::front::wgsl::parse_str(source).map_err(super::invalid)?;
    let mut result = Vec::new();
    for function in module
        .functions
        .iter()
        .map(|(_, f)| f)
        .chain(module.entry_points.iter().map(|e| &e.function))
    {
        let mut ranges = Vec::new();
        let mut stores = Vec::new();
        walk(&function.body, &mut ranges, &mut stores);
        for (handle, local) in function.local_variables.iter() {
            if local.name.is_none() || local.init.is_some() {
                continue;
            }
            let Some(span) = function.local_variables.get_span(handle).to_range() else {
                continue;
            };
            if stores.iter().any(|(pointer, store_span)| {
                store_span.to_range() == Some(span.clone())
                    && matches!(function.expressions[*pointer], naga::Expression::LocalVariable(h) if h == handle)
            }) {
                continue;
            }
            if ranges
                .iter()
                .any(|r| r.start <= span.start && span.end <= r.end)
            {
                result.push(Local {
                    function: function.name.clone().unwrap_or_default(),
                    name: local.name.clone().unwrap_or_default(),
                    declaration: span,
                });
            }
        }
    }
    result.sort_by_key(|l| l.declaration.start);
    Ok(result)
}

#[test]
fn newton_loop_local_audit_distinguishes_declaration_stores() {
    let source = r#"
fn f(input:u32) -> u32 {
    var outside:u32;
    for(var i=0u;i<2u;i++) {
        var implicit /* = /* nested */ */ :u32;
        implicit=4u;
        var initialized:u32=0u;
        var dynamic:u32=input;
        var inferred=input;
        if input>0u && i>0u {var branch:u32;outside+=branch;}
        switch i {case 0u: {var switched:u32;outside+=switched;} default: {}}
        outside+=implicit+initialized+dynamic+inferred;
    }
    loop {var repeated:u32;outside+=repeated;break;}
    return outside;
}
"#;
    let locals = audit(source).unwrap();
    let names: Vec<_> = locals.iter().map(|l| l.name.as_str()).collect();
    assert_eq!(names, ["implicit", "branch", "switched", "repeated"]);
    for local in locals {
        assert!(source[local.declaration].starts_with("var "));
    }
}

#[test]
fn newton_loop_local_audit() {
    use crate::soil_newton_gpu::{
        ExperimentalSoilNewtonGpu as Solver, NewtonVariant, shader_source_for,
    };
    let source = crate::soil_newton_parallel_release::schur_inverse::source(
        crate::soil_newton_parallel_release::core_source(
            crate::soil_newton_parallel_release::source(Solver::cached_contact_source(
                Solver::cached_delta_source(Solver::stalled_release_source(shader_source_for(
                    NewtonVariant::Fischer,
                ))),
            )),
        ),
    );
    let locals = audit(&source).unwrap();
    assert!(!locals.is_empty());
    let mut output = String::new();
    for local in locals {
        output.push_str(&format!(
            "{}::{} [{}..{}] {}\n",
            local.function,
            local.name,
            local.declaration.start,
            local.declaration.end,
            &source[local.declaration.clone()]
        ));
    }
    eprint!("{output}");
    if let Some(root) = std::env::var_os("MOLLA_NEWTON_LOCAL_AUDIT") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("locals.txt"), output).unwrap();
        std::fs::write(root.join("source.wgsl"), source).unwrap();
    }
}
