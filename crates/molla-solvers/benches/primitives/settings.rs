type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub const HELP: &str = "--primitive all|distance|cone|d6|reaction|break|friction|mixed\n--counts 1,100,1000,10000 --backend both|cpu|gpu --warmup 20 --steps 1000 --dt 0.004166666666666667 --substeps 4 --case-timeout-seconds 3600 [--trace-frames]\n--bench selects the full performance defaults; --smoke or --test selects one instance, one warmup and two measured steps. Cargo test defaults to smoke mode.";

#[derive(Debug)]
pub struct Settings {
    pub worker: bool,
    pub smoke: bool,
    pub primitive: String,
    pub counts: Vec<usize>,
    pub backend: String,
    pub warmup: usize,
    pub steps: usize,
    pub dt: f64,
    pub substeps: usize,
    pub case_timeout_seconds: u64,
    pub trace_frames: bool,
}

impl Settings {
    pub fn parse(args: impl IntoIterator<Item = String>, cases: &[&str]) -> Result<Option<Self>> {
        let args: Vec<_> = args.into_iter().collect();
        if args.iter().any(|arg| arg == "--help") {
            return Ok(None);
        }
        let worker = args.iter().any(|arg| arg == "--worker");
        let benchmark = args.iter().any(|arg| arg == "--bench");
        let smoke =
            args.iter().any(|arg| arg == "--smoke" || arg == "--test") || (!benchmark && !worker);
        let mut settings = Self {
            worker,
            smoke,
            primitive: "all".into(),
            counts: if smoke {
                vec![1]
            } else {
                vec![1, 100, 1000, 10000]
            },
            backend: "both".into(),
            warmup: if smoke { 1 } else { 20 },
            steps: if smoke { 2 } else { 1000 },
            dt: 1. / 240.,
            substeps: 4,
            case_timeout_seconds: 3600,
            trace_frames: false,
        };
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--worker" | "--bench" | "--smoke" | "--test" => continue,
                "--trace-frames" => {
                    settings.trace_frames = true;
                    continue;
                }
                _ => {}
            }
            let value = args.next().ok_or("missing option value")?;
            match arg.as_str() {
                "--primitive" => settings.primitive = value,
                "--counts" => {
                    settings.counts = value
                        .split(',')
                        .map(str::parse)
                        .collect::<std::result::Result<_, _>>()?
                }
                "--backend" => settings.backend = value,
                "--warmup" => settings.warmup = value.parse()?,
                "--steps" => settings.steps = value.parse()?,
                "--dt" => settings.dt = value.parse()?,
                "--substeps" => settings.substeps = value.parse()?,
                "--case-timeout-seconds" => settings.case_timeout_seconds = value.parse()?,
                _ => return Err(format!("unknown option {arg}").into()),
            }
        }
        if (settings.primitive != "all" && !cases.contains(&settings.primitive.as_str()))
            || !["both", "cpu", "gpu"].contains(&settings.backend.as_str())
            || settings.counts.is_empty()
            || settings.counts.contains(&0)
            || settings.steps == 0
            || settings.warmup.checked_add(settings.steps).is_none()
            || settings.substeps == 0
            || settings.case_timeout_seconds == 0
            || !settings.dt.is_finite()
            || settings.dt <= 0.
        {
            return Err("invalid benchmark settings".into());
        }
        if worker
            && (settings.primitive == "all"
                || settings.counts.len() != 1
                || settings.backend == "both")
        {
            return Err("benchmark worker requires one case and backend".into());
        }
        Ok(Some(settings))
    }
}
