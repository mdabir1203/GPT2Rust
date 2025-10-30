use std::env;
use std::ffi::CString;
use std::fs::{self, OpenOptions};
use std::io::{self, Error, ErrorKind, Read, Write};
use std::os::fd::RawFd;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::ptr;
use std::time::{SystemTime, UNIX_EPOCH};

const STACK_SIZE: usize = 1024 * 1024;

type Result<T> = std::result::Result<T, io::Error>;

mod linux {
    use std::os::raw::{c_char, c_int, c_ulong, c_void};

    pub type Pid = c_int;
    pub const SIGCHLD: c_int = 17;
    pub const CLONE_NEWUTS: c_int = 0x0400_0000;
    pub const CLONE_NEWIPC: c_int = 0x0800_0000;
    pub const CLONE_NEWNS: c_int = 0x0002_0000;
    pub const CLONE_NEWPID: c_int = 0x2000_0000;
    pub const CLONE_NEWNET: c_int = 0x4000_0000;

    pub const MS_BIND: c_ulong = 4096;
    pub const MS_REC: c_ulong = 16384;
    pub const MS_PRIVATE: c_ulong = 262144;
    pub const MS_NOSUID: c_ulong = 2;
    pub const MS_NODEV: c_ulong = 4;
    pub const MS_NOEXEC: c_ulong = 8;

    extern "C" {
        pub fn clone(
            f: extern "C" fn(*mut c_void) -> c_int,
            child_stack: *mut c_void,
            flags: c_int,
            arg: *mut c_void,
        ) -> c_int;
        pub fn pipe(fds: *mut c_int) -> c_int;
        pub fn close(fd: c_int) -> c_int;
        pub fn read(fd: c_int, buf: *mut c_void, count: usize) -> isize;
        pub fn write(fd: c_int, buf: *const c_void, count: usize) -> isize;
        pub fn waitpid(pid: c_int, status: *mut c_int, options: c_int) -> c_int;
        pub fn sethostname(name: *const c_char, len: usize) -> c_int;
        pub fn chdir(path: *const c_char) -> c_int;
        pub fn chroot(path: *const c_char) -> c_int;
        pub fn mount(
            source: *const c_char,
            target: *const c_char,
            fstype: *const c_char,
            flags: c_ulong,
            data: *const c_void,
        ) -> c_int;
        pub fn execvp(file: *const c_char, argv: *const *const c_char) -> c_int;
    }

    pub unsafe fn wifexited(status: c_int) -> bool {
        status & 0x7f == 0
    }

    pub unsafe fn wexitstatus(status: c_int) -> c_int {
        (status >> 8) & 0xff
    }

    pub unsafe fn wifsignaled(status: c_int) -> bool {
        let sig = status & 0x7f;
        sig != 0 && sig != 0x7f
    }

    pub unsafe fn wtermsig(status: c_int) -> c_int {
        status & 0x7f
    }
}

#[derive(Clone, Debug)]
struct Config {
    rootfs: PathBuf,
    memory: Option<u64>,
    cpu: Option<u32>,
    hostname: String,
    command: Vec<String>,
}

#[derive(Clone, Debug)]
struct ChildConfig {
    rootfs: PathBuf,
    command: Vec<String>,
    hostname: String,
    sync_read: RawFd,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("mini-docker error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cfg = parse_args()?;
    if !cfg.rootfs.is_dir() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!("rootfs '{}' must exist", cfg.rootfs.display()),
        ));
    }
    if cfg.command.is_empty() {
        return Err(Error::new(ErrorKind::InvalidInput, "no command specified"));
    }

    let mut pipe_fds = [0; 2];
    check(unsafe { linux::pipe(pipe_fds.as_mut_ptr()) }, "pipe")?;
    let (read_fd, write_fd) = (pipe_fds[0], pipe_fds[1]);
    let mut stack = vec![0u8; STACK_SIZE];
    let child_cfg = Box::new(ChildConfig {
        rootfs: cfg.rootfs.clone(),
        command: cfg.command.clone(),
        hostname: cfg.hostname.clone(),
        sync_read: read_fd,
    });

    let flags = linux::CLONE_NEWUTS
        | linux::CLONE_NEWPID
        | linux::CLONE_NEWNS
        | linux::CLONE_NEWIPC
        | linux::CLONE_NEWNET
        | linux::SIGCHLD;
    let ptr = Box::into_raw(child_cfg) as *mut std::ffi::c_void;
    let stack_top = unsafe { stack.as_mut_ptr().add(STACK_SIZE) } as *mut std::ffi::c_void;
    let pid = unsafe { linux::clone(child_trampoline, stack_top, flags, ptr) };
    if pid < 0 {
        unsafe {
            drop(Box::from_raw(ptr as *mut ChildConfig));
        }
        return Err(last_error("clone"));
    }

    let controller = match Cgroup::new(cfg.memory, cfg.cpu) {
        Ok(cg) => cg,
        Err(err) => {
            unsafe {
                let _ = linux::close(read_fd);
                let _ = linux::close(write_fd);
            }
            let _ = wait_for_child(pid);
            return Err(err);
        }
    };
    controller.attach(pid)?;
    check(unsafe { linux::close(read_fd) }, "close")?;
    write_all(write_fd, &[1])?;
    check(unsafe { linux::close(write_fd) }, "close")?;
    wait_for_child(pid)
}

fn parse_args() -> Result<Config> {
    let mut args = env::args().skip(1).peekable();
    let mut rootfs = None;
    let mut memory = None;
    let mut cpu = None;
    let mut hostname = String::from("mini-docker");
    let mut command = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--rootfs" => rootfs = Some(PathBuf::from(expect_value(&mut args, "--rootfs")?)),
            "--memory" => {
                let value = expect_value(&mut args, "--memory")?;
                memory =
                    Some(parse_memory(&value).map_err(|e| Error::new(ErrorKind::InvalidInput, e))?);
            }
            "--cpu" => {
                let value = expect_value(&mut args, "--cpu")?;
                cpu = Some(
                    parse_cpu_percent(&value)
                        .map_err(|e| Error::new(ErrorKind::InvalidInput, e))?,
                );
            }
            "--hostname" => hostname = expect_value(&mut args, "--hostname")?,
            "--" => {
                command.extend(args);
                break;
            }
            other if other.starts_with("--") => {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    format!("unknown option {other}"),
                ))
            }
            _ => {
                command.push(arg);
                command.extend(args);
                break;
            }
        }
    }
    let rootfs =
        rootfs.ok_or_else(|| Error::new(ErrorKind::InvalidInput, "--rootfs is required"))?;
    Ok(Config {
        rootfs,
        memory,
        cpu,
        hostname,
        command,
    })
}

fn expect_value(
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    flag: &str,
) -> Result<String> {
    args.next()
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, format!("{flag} requires a value")))
}

fn path_to_cstring(path: &Path) -> Result<CString> {
    CString::new(path.as_os_str().as_bytes()).map_err(|_| {
        Error::new(
            ErrorKind::InvalidInput,
            format!("path '{}' contains an embedded null byte", path.display()),
        )
    })
}

extern "C" fn child_trampoline(arg: *mut std::ffi::c_void) -> i32 {
    unsafe {
        let cfg = Box::from_raw(arg as *mut ChildConfig);
        let status = match child_main(&cfg) {
            Ok(()) => 0,
            Err(err) => {
                eprintln!("[mini-docker] child error: {err}");
                1
            }
        };
        let _ = linux::close(cfg.sync_read);
        status
    }
}

fn child_main(cfg: &ChildConfig) -> Result<()> {
    set_hostname(&cfg.hostname)?;
    let mut buf = [0u8; 1];
    read_all(cfg.sync_read, &mut buf)?;
    check(unsafe { linux::close(cfg.sync_read) }, "close")?;
    setup_rootfs(&cfg.rootfs)?;
    exec_command(&cfg.command)
}

fn set_hostname(name: &str) -> Result<()> {
    let c =
        CString::new(name).map_err(|_| Error::new(ErrorKind::InvalidInput, "invalid hostname"))?;
    check(
        unsafe { linux::sethostname(c.as_ptr(), name.len()) },
        "sethostname",
    )
}

fn setup_rootfs(rootfs: &Path) -> Result<()> {
    for dir in ["proc", "sys", "tmp"] {
        let path = rootfs.join(dir);
        if !path.exists() {
            fs::create_dir_all(&path)?;
        }
    }
    mount_call(
        None,
        Path::new("/"),
        None,
        linux::MS_REC | linux::MS_PRIVATE,
        None,
    )?;
    bind_mount(rootfs, rootfs)?;
    chdir(rootfs)?;
    chroot_to_current()?;
    chdir(Path::new("/"))?;
    mount_call(
        Some("proc"),
        Path::new("/proc"),
        Some("proc"),
        linux::MS_NOSUID | linux::MS_NOEXEC | linux::MS_NODEV,
        None,
    )?;
    mount_call(
        Some("sysfs"),
        Path::new("/sys"),
        Some("sysfs"),
        linux::MS_NOSUID | linux::MS_NOEXEC | linux::MS_NODEV,
        None,
    )?;
    mount_call(
        Some("tmpfs"),
        Path::new("/tmp"),
        Some("tmpfs"),
        0,
        Some("mode=1777"),
    )
}

fn mount_call(
    source: Option<&str>,
    target: &Path,
    fstype: Option<&str>,
    flags: u64,
    data: Option<&str>,
) -> Result<()> {
    let src = source.map(|s| CString::new(s).unwrap());
    let tgt = path_to_cstring(target)?;
    let fstype = fstype.map(|f| CString::new(f).unwrap());
    let data_c = data.map(|d| CString::new(d).unwrap());
    let result = unsafe {
        linux::mount(
            src.as_ref().map(|s| s.as_ptr()).unwrap_or(ptr::null()),
            tgt.as_ptr(),
            fstype.as_ref().map(|f| f.as_ptr()).unwrap_or(ptr::null()),
            flags,
            data_c
                .as_ref()
                .map(|d| d.as_ptr() as *const std::ffi::c_void)
                .unwrap_or(ptr::null()),
        )
    };
    check(result, "mount")
}

fn bind_mount(source: &Path, target: &Path) -> Result<()> {
    let src = path_to_cstring(source)?;
    let tgt = path_to_cstring(target)?;
    let result = unsafe {
        linux::mount(
            src.as_ptr(),
            tgt.as_ptr(),
            ptr::null(),
            linux::MS_BIND | linux::MS_REC,
            ptr::null(),
        )
    };
    check(result, "bind mount")
}

fn chdir(path: &Path) -> Result<()> {
    let c = path_to_cstring(path)?;
    check(unsafe { linux::chdir(c.as_ptr()) }, "chdir")
}

fn chroot_to_current() -> Result<()> {
    let dot = CString::new(".").unwrap();
    check(unsafe { linux::chroot(dot.as_ptr()) }, "chroot")
}

fn exec_command(args: &[String]) -> Result<()> {
    if args.is_empty() {
        return Err(Error::new(ErrorKind::InvalidInput, "no command to exec"));
    }
    let mut c_args = Vec::with_capacity(args.len());
    for arg in args {
        c_args.push(
            CString::new(arg.as_str())
                .map_err(|_| Error::new(ErrorKind::InvalidInput, "invalid string"))?,
        );
    }
    let mut raw: Vec<*const std::ffi::c_char> = c_args.iter().map(|s| s.as_ptr()).collect();
    raw.push(ptr::null());
    let status = unsafe { linux::execvp(raw[0], raw.as_ptr()) };
    if status == -1 {
        Err(last_error("execvp"))
    } else {
        Ok(())
    }
}

fn wait_for_child(pid: linux::Pid) -> Result<()> {
    let mut status = 0;
    check(unsafe { linux::waitpid(pid, &mut status, 0) }, "waitpid")?;
    if unsafe { linux::wifexited(status) } {
        match unsafe { linux::wexitstatus(status) } {
            0 => Ok(()),
            code => Err(Error::new(
                ErrorKind::Other,
                format!("container exited with status {code}"),
            )),
        }
    } else if unsafe { linux::wifsignaled(status) } {
        Err(Error::new(
            ErrorKind::Other,
            format!("container terminated by signal {}", unsafe {
                linux::wtermsig(status)
            }),
        ))
    } else {
        Err(Error::new(ErrorKind::Other, "unexpected wait status"))
    }
}

fn read_all(fd: RawFd, buf: &mut [u8]) -> Result<()> {
    let mut offset = 0;
    while offset < buf.len() {
        let res = unsafe {
            linux::read(
                fd,
                buf[offset..].as_mut_ptr() as *mut std::ffi::c_void,
                buf.len() - offset,
            )
        };
        if res == -1 {
            let err = last_error("read");
            if err.kind() == ErrorKind::Interrupted {
                continue;
            }
            return Err(err);
        }
        if res == 0 {
            return Err(Error::new(ErrorKind::UnexpectedEof, "pipe closed"));
        }
        offset += res as usize;
    }
    Ok(())
}

fn write_all(fd: RawFd, buf: &[u8]) -> Result<()> {
    let mut offset = 0;
    while offset < buf.len() {
        let res = unsafe {
            linux::write(
                fd,
                buf[offset..].as_ptr() as *const std::ffi::c_void,
                buf.len() - offset,
            )
        };
        if res == -1 {
            let err = last_error("write");
            if err.kind() == ErrorKind::Interrupted {
                continue;
            }
            return Err(err);
        }
        offset += res as usize;
    }
    Ok(())
}

fn check(result: i32, action: &str) -> Result<()> {
    if result == -1 {
        Err(last_error(action))
    } else {
        Ok(())
    }
}

fn last_error(action: &str) -> io::Error {
    let err = io::Error::last_os_error();
    Error::new(err.kind(), format!("{action}: {err}"))
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
            return Err(Error::new(ErrorKind::Other, "cgroup v2 is required"));
        }
        if memory.is_some() {
            enable_controller(&base, "memory")?;
        }
        if cpu.is_some() {
            enable_controller(&base, "cpu")?;
        }
        let path = base.join(format!("mini-docker-{}", unique_suffix()));
        fs::create_dir_all(&path)?;
        let cg = Self { path, memory, cpu };
        cg.apply_limits()?;
        Ok(cg)
    }

    fn apply_limits(&self) -> Result<()> {
        if let Some(bytes) = self.memory {
            write_file(self.path.join("memory.max"), bytes.to_string())?;
        }
        if let Some(percent) = self.cpu {
            let period = 100_000u64;
            let quota = (period * percent as u64).max(100) / 100;
            write_file(self.path.join("cpu.max"), format!("{quota} {period}"))?;
        }
        Ok(())
    }

    fn attach(&self, pid: linux::Pid) -> Result<()> {
        write_file(self.path.join("cgroup.procs"), pid.to_string())
    }
}

impl Drop for Cgroup {
    fn drop(&mut self) {
        let _ = write_file(self.path.join("cpu.max"), "max 100000");
        let _ = write_file(self.path.join("memory.max"), "max");
        let _ = fs::remove_dir(&self.path);
    }
}

fn write_file(path: PathBuf, data: impl AsRef<[u8]>) -> Result<()> {
    let mut file = OpenOptions::new().write(true).truncate(true).open(&path)?;
    file.write_all(data.as_ref())?;
    Ok(())
}

fn enable_controller(base: &Path, controller: &str) -> Result<()> {
    let control_file = base.join("cgroup.subtree_control");
    let mut existing = String::new();
    if let Ok(mut file) = OpenOptions::new().read(true).open(&control_file) {
        file.read_to_string(&mut existing)?;
    }
    if !existing
        .split_whitespace()
        .any(|item| item.trim_start_matches('+') == controller)
    {
        let mut file = OpenOptions::new().append(true).open(&control_file)?;
        file.write_all(format!("+{}\n", controller).as_bytes())?;
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
    if trimmed.is_empty() {
        return Err("memory size cannot be empty".into());
    }
    let (number, suffix) = trimmed
        .chars()
        .position(|c| !c.is_ascii_digit())
        .map(|idx| trimmed.split_at(idx))
        .unwrap_or((trimmed.as_str(), ""));
    let value: u64 = number
        .parse()
        .map_err(|_| format!("invalid number: {input}"))?;
    let multiplier = match suffix {
        "" => 1,
        "k" | "kb" => 1 << 10,
        "m" | "mb" => 1 << 20,
        "g" | "gb" => 1 << 30,
        _ => return Err(format!("unknown size suffix: {suffix}")),
    };
    Ok(value.saturating_mul(multiplier))
}

fn parse_cpu_percent(input: &str) -> std::result::Result<u32, String> {
    let value: u32 = input
        .parse()
        .map_err(|_| format!("invalid cpu percentage: {input}"))?;
    if (1..=100).contains(&value) {
        Ok(value)
    } else {
        Err("cpu percentage must be between 1 and 100".into())
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_cpu_percent, parse_memory, unique_suffix};

    #[test]
    fn parse_memory_handles_common_suffixes() {
        assert_eq!(parse_memory("1024").unwrap(), 1024);
        assert_eq!(parse_memory("1k").unwrap(), 1024);
        assert_eq!(parse_memory("2M").unwrap(), 2 * 1024 * 1024);
        assert_eq!(parse_memory("3Gb").unwrap(), 3 * 1024 * 1024 * 1024);
    }

    #[test]
    fn parse_memory_rejects_invalid_inputs() {
        assert!(parse_memory("abc").is_err());
        assert!(parse_memory("12tb").is_err());
        assert!(parse_memory("").is_err());
    }

    #[test]
    fn parse_cpu_percent_checks_bounds() {
        assert_eq!(parse_cpu_percent("1").unwrap(), 1);
        assert_eq!(parse_cpu_percent("100").unwrap(), 100);
        assert!(parse_cpu_percent("0").is_err());
        assert!(parse_cpu_percent("101").is_err());
    }

    #[test]
    fn unique_suffix_monotonicity() {
        let first = unique_suffix();
        let second = unique_suffix();
        assert!(first > 0);
        assert!(second >= first);
    }
}
