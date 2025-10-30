use std::ffi::CString;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::os::fd::RawFd;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, ensure, Context, Result};
use clap::Parser;
use nix::mount::{mount, MsFlags};
use nix::sched::{clone, CloneFlags};
use nix::sys::signal::Signal;
use nix::sys::wait::{waitpid, WaitStatus};
use nix::unistd::{chdir, chroot, close, execvp, pipe, read, sethostname, write as nix_write, Pid};

const STACK_SIZE: usize = 1024 * 1024;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "mini-docker",
    about = "A mini container runtime in ~300 lines of Rust"
)]
struct Config {
    #[arg(long, value_name = "PATH")]
    rootfs: PathBuf,
    #[arg(long, value_name = "SIZE", value_parser = parse_memory)]
    memory: Option<u64>,
    #[arg(long, value_name = "PERCENT", value_parser = parse_cpu_percent)]
    cpu: Option<u32>,
    #[arg(long, default_value = "mini-docker")]
    hostname: String,
    #[arg(required = true, trailing_var_arg = true)]
    command: Vec<String>,
}

fn main() -> Result<()> {
    let cfg = Config::parse();
    ensure!(
        cfg.rootfs.is_dir(),
        "rootfs '{}' must be an existing directory",
        cfg.rootfs.display()
    );
    ensure!(!cfg.command.is_empty(), "no command specified");
    run_container(cfg)
}

fn run_container(cfg: Config) -> Result<()> {
    let (sync_read, sync_write) = pipe().context("creating sync pipe")?;
    let mut stack = vec![0u8; STACK_SIZE];
    let child_cfg = ChildConfig(
        cfg.rootfs.clone(),
        cfg.command.clone(),
        cfg.hostname.clone(),
        sync_read,
    );
    let flags = CloneFlags::CLONE_NEWUTS
        | CloneFlags::CLONE_NEWPID
        | CloneFlags::CLONE_NEWNS
        | CloneFlags::CLONE_NEWIPC
        | CloneFlags::CLONE_NEWNET;
    let child_pid = unsafe {
        clone(
            Box::new(move || child_entry(child_cfg.clone())),
            &mut stack,
            flags,
            Some(Signal::SIGCHLD as i32),
        )
    }
    .context("failed to clone child")?;
    close(sync_read).ok();

    let controller = Cgroup::new(cfg.memory, cfg.cpu).context("setting up cgroup")?;
    controller
        .attach(child_pid)
        .context("attaching child to cgroup")?;
    nix_write(sync_write, &[1]).context("signalling child")?;
    close(sync_write).ok();

    match waitpid(child_pid, None).context("waiting for container")? {
        WaitStatus::Exited(_, code) if code == 0 => {}
        WaitStatus::Exited(_, code) => {
            return Err(anyhow!("container exited with status {}", code))
        }
        WaitStatus::Signaled(_, sig, _) => {
            return Err(anyhow!("container terminated by signal {:?}", sig))
        }
        other => return Err(anyhow!("unexpected wait status: {:?}", other)),
    }
    drop(controller);
    Ok(())
}

#[derive(Clone, Debug)]
struct ChildConfig(PathBuf, Vec<String>, String, RawFd);

fn child_entry(config: ChildConfig) -> isize {
    match child_main(&config) {
        Ok(()) => 0,
        Err(err) => {
            eprintln!("[mini-docker] child error: {err:?}");
            1
        }
    }
}

fn child_main(config: &ChildConfig) -> Result<()> {
    sethostname(config.2.as_str()).context("setting hostname")?;
    let mut buf = [0u8; 1];
    read(config.3, &mut buf).context("waiting for parent")?;
    close(config.3).ok();
    setup_rootfs(&config.0).context("configuring filesystem")?;
    let argv: Vec<CString> = config
        .1
        .iter()
        .map(|arg| CString::new(arg.as_str()).map_err(|_| anyhow!("invalid string: {arg}")))
        .collect::<Result<_>>()?;
    if argv.is_empty() {
        return Err(anyhow!("no command to exec"));
    }
    execvp(&argv[0], &argv).context("execvp failed")?;
    Ok(())
}

fn setup_rootfs(rootfs: &Path) -> Result<()> {
    for dir in ["proc", "sys", "tmp"] {
        let path = rootfs.join(dir);
        if !path.exists() {
            fs::create_dir_all(&path).with_context(|| format!("creating {}", path.display()))?;
        }
    }
    mount::<str, Path, str, str>(
        None,
        Path::new("/"),
        None,
        MsFlags::MS_REC | MsFlags::MS_PRIVATE,
        None,
    )
    .context("remount private")?;
    mount(
        Some(rootfs),
        rootfs,
        None::<&str>,
        MsFlags::MS_BIND | MsFlags::MS_REC,
        None::<&str>,
    )
    .with_context(|| format!("binding {}", rootfs.display()))?;
    chdir(rootfs).context("chdir into rootfs")?;
    chroot(".").context("chroot")?;
    chdir(Path::new("/")).context("enter new root")?;
    mount(
        Some("proc"),
        Path::new("/proc"),
        Some("proc"),
        MsFlags::MS_NOSUID | MsFlags::MS_NOEXEC | MsFlags::MS_NODEV,
        None::<&str>,
    )
    .context("mount proc")?;
    mount(
        Some("sysfs"),
        Path::new("/sys"),
        Some("sysfs"),
        MsFlags::MS_NOSUID | MsFlags::MS_NOEXEC | MsFlags::MS_NODEV,
        None::<&str>,
    )
    .context("mount sysfs")?;
    mount(
        Some("tmpfs"),
        Path::new("/tmp"),
        Some("tmpfs"),
        MsFlags::empty(),
        Some("mode=1777"),
    )
    .context("mount tmpfs")?;
    Ok(())
}

struct Cgroup {
    path: PathBuf,
    memory: Option<u64>,
    cpu: Option<u32>,
}

impl Cgroup {
    fn new(memory: Option<u64>, cpu: Option<u32>) -> Result<Self> {
        let base = PathBuf::from("/sys/fs/cgroup");
        if !base.join("cgroup.controllers").exists() {
            return Err(anyhow!("cgroup v2 is required"));
        }
        if memory.is_some() {
            enable_controller(&base, "memory").ok();
        }
        if cpu.is_some() {
            enable_controller(&base, "cpu").ok();
        }
        let path = base.join(format!("mini-docker-{}", unique_suffix()));
        fs::create_dir_all(&path).with_context(|| format!("creating cgroup {}", path.display()))?;
        let cg = Self { path, memory, cpu };
        cg.apply_limits()?;
        Ok(cg)
    }

    fn apply_limits(&self) -> Result<()> {
        if let Some(bytes) = self.memory {
            let value = bytes.to_string();
            write_file(self.path.join("memory.max"), value.as_bytes())?;
        }
        if let Some(percent) = self.cpu {
            let period: u64 = 100_000;
            let quota = (period * percent as u64).max(100) / 100;
            let data = format!("{} {}", quota, period);
            write_file(self.path.join("cpu.max"), data.as_bytes())?;
        }
        Ok(())
    }

    fn attach(&self, pid: Pid) -> Result<()> {
        write_file(
            self.path.join("cgroup.procs"),
            pid.as_raw().to_string().as_bytes(),
        )
    }
}

impl Drop for Cgroup {
    fn drop(&mut self) {
        let _ = write_file(self.path.join("cpu.max"), b"max 100000");
        let _ = write_file(self.path.join("memory.max"), b"max");
        let _ = fs::remove_dir(&self.path);
    }
}

fn write_file(path: PathBuf, data: &[u8]) -> Result<()> {
    OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(&path)
        .with_context(|| format!("opening {}", path.display()))?
        .write_all(data)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

fn enable_controller(base: &Path, controller: &str) -> Result<()> {
    let control_file = base.join("cgroup.subtree_control");
    let mut current = String::new();
    if let Ok(mut existing) = OpenOptions::new().read(true).open(&control_file) {
        existing.read_to_string(&mut current).ok();
    }
    if !current
        .split_whitespace()
        .any(|c| c.trim_start_matches('+') == controller)
    {
        OpenOptions::new()
            .append(true)
            .open(&control_file)
            .with_context(|| format!("opening {}", control_file.display()))?
            .write_all(format!("+{}\n", controller).as_bytes())
            .with_context(|| format!("enabling controller {}", controller))?;
    }
    Ok(())
}

fn unique_suffix() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or_default()
}

fn parse_memory(input: &str) -> std::result::Result<u64, String> {
    let trimmed = input.trim().to_ascii_lowercase();
    let (number, suffix) = trimmed
        .chars()
        .position(|c| !c.is_ascii_digit())
        .map(|idx| trimmed.split_at(idx))
        .unwrap_or((trimmed.as_str(), ""));
    let value: u64 = number
        .parse()
        .map_err(|_| format!("invalid number: {}", input))?;
    let multiplier = match suffix {
        "" => 1,
        "k" | "kb" => 1 << 10,
        "m" | "mb" => 1 << 20,
        "g" | "gb" => 1 << 30,
        _ => return Err(format!("unknown size suffix: {}", suffix)),
    };
    Ok(value.saturating_mul(multiplier))
}

fn parse_cpu_percent(input: &str) -> std::result::Result<u32, String> {
    let value: u32 = input
        .parse()
        .map_err(|_| format!("invalid cpu percentage: {}", input))?;
    if (1..=100).contains(&value) {
        Ok(value)
    } else {
        Err("cpu percentage must be between 1 and 100".into())
    }
}
