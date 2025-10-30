# Mini Docker in Rust

This repository contains a pedagogical, single-binary container runtime that
reimplements a subset of Docker's process-isolation features in roughly three
hundred lines of Rust. The program uses Linux namespaces to create an isolated
UTS/PID/IPC/NET/MOUNT environment, wires the process into a dedicated cgroup v2
hierarchy, then pivots into a BusyBox-style root filesystem before executing the
requested command.

The runtime favours readability and explicit error handling so that you can
trace every syscall. This README concentrates on verifying that the runtime
actually provides isolation and resource-limiting guarantees, and documents the
unit tests that keep the Rust code honest.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Building](#building)
- [Test Matrix Overview](#test-matrix-overview)
- [Preparing a Minimal Root Filesystem](#preparing-a-minimal-root-filesystem)
- [Happy-Path Container Execution](#happy-path-container-execution)
- [Resource Limit Validation](#resource-limit-validation)
  - [Memory Pressure Scenarios](#memory-pressure-scenarios)
  - [CPU Throttling Scenarios](#cpu-throttling-scenarios)
- [Filesystem Isolation Checks](#filesystem-isolation-checks)
- [Networking Containment Checks](#networking-containment-checks)
- [Failure Mode Drills](#failure-mode-drills)
- [Rust Unit Tests](#rust-unit-tests)
- [Troubleshooting Checklist](#troubleshooting-checklist)
- [Further Exploration](#further-exploration)

## Prerequisites

| Requirement | Why it matters | How to verify |
|-------------|----------------|----------------|
| Linux kernel with namespace & cgroup v2 support | All isolation primitives depend on them. | `ls /proc/self/ns && ls /sys/fs/cgroup/cgroup.controllers` |
| Root or a user with `CAP_SYS_ADMIN` | Mount, chroot, and cgroup operations require elevated privileges. | `id -u` should print `0` or verify capabilities with `capsh --print`. |
| BusyBox (or equivalent minimal rootfs) | Provides shell utilities inside the container. | `busybox --help` |
| Rust toolchain (`cargo`, `rustc`) | Builds the runtime and runs unit tests. | `cargo --version` |

Mental model: treat the runtime as an orchestra conductor. Each prerequisite is
an instrument that must be tuned before the performance can start.

## Building

1. Fetch dependencies and compile in release mode for predictable behaviour:

   ```bash
   cargo build --release
   ```

2. Optionally run `cargo fmt` to ensure formatting consistency. The code is
   deliberately compact, so formatting aids future diffs.

3. Confirm the binary exists:

   ```bash
   file target/release/mini-docker
   ```

Think of this phase as setting the stage—no tests should run until the binary is
known-good.

## Test Matrix Overview

Before diving into shell commands, align expectations using the following
matrix. It groups scenarios by goal and helps prioritise what to test first.

| Goal | Mental Model | Primary Check | Edge Cases |
|------|--------------|---------------|------------|
| Baseline execution | "Does the light turn on?" | Container launches BusyBox shell | Missing rootfs, invalid command |
| Memory limits | "Turn the dial until the fuse blows." | Process killed when exceeding limit | Child ignoring limit, swap usage |
| CPU limits | "Hourglass throttle." | Loop slows under capped quota | Multi-threaded workloads |
| Filesystem isolation | "Clean-room lab." | Host files hidden inside container | Bind mounts, tmpfs persistence |
| Networking isolation | "Airplane mode." | Container can't reach external hosts | Localhost-only services |
| Failure handling | "Fire drill." | Clear error messages & cleanup | Partial setup, repeated runs |

## Preparing a Minimal Root Filesystem

1. Create a workspace and populate it with BusyBox:

   ```bash
   ROOTFS=$PWD/rootfs
   mkdir -p "$ROOTFS"
   busybox --install -s "$ROOTFS/bin"
   mkdir -p "$ROOTFS"/{dev,proc,sys,tmp,etc}
   printf 'root:x:0:0:root:/root:/bin/sh\n' > "$ROOTFS/etc/passwd"
   printf 'nameserver 1.1.1.1\n' > "$ROOTFS/etc/resolv.conf"
   ```

2. Add essential device nodes (requires root):

   ```bash
   mknod -m 666 "$ROOTFS/dev/null" c 1 3
   mknod -m 666 "$ROOTFS/dev/zero" c 1 5
   mknod -m 666 "$ROOTFS/dev/random" c 1 8
   mknod -m 666 "$ROOTFS/dev/urandom" c 1 9
   ```

3. Verify layout:

   ```bash
   tree -L 2 "$ROOTFS"
   ```

Mental model: build a dollhouse—every room (directory) must exist before the
runtime can move in furniture (mounts).

## Happy-Path Container Execution

```bash
sudo target/release/mini-docker \
  --rootfs "$ROOTFS" \
  --hostname demo \
  /bin/sh -lc "echo hello from $(hostname) && ls /"
```

Expected outcomes:

- Prompt prints `hello from demo` showing the UTS namespace works.
- `/` listing shows BusyBox directories rather than the host root.
- Exiting the shell should terminate the container cleanly with exit code `0`.

Edge cases to probe immediately:

- Launch with a missing command (`-- rootfs "$ROOTFS"`) to ensure the CLI rejects
  it.
- Point `--rootfs` at a non-directory path and confirm the binary errors with a
  helpful message.

## Resource Limit Validation

### Memory Pressure Scenarios

1. Run a stress test within a strict limit:

   ```bash
   sudo target/release/mini-docker \
     --rootfs "$ROOTFS" \
     --memory 32m \
     /bin/sh -lc "echo $$ > /tmp/pid; stress-ng --vm 1 --vm-bytes 64m --timeout 20"
   ```

2. Observe behaviour:

   - `stress-ng` should be terminated by the kernel (`dmesg | tail`) with an OOM
     message referencing the container PID.
   - The runtime should exit non-zero and print that the container exited with a
     status reflecting the kill.

3. Edge cases:

   - Repeat with `--memory 0` (invalid) and ensure argument parsing rejects it.
   - Try `--memory 4k` to confirm suffix handling.

Mental model: treat the cgroup like a circuit breaker—push until it trips and
verify the lights go out without burning the house down.

### CPU Throttling Scenarios

1. Launch a busy loop capped at 20% CPU:

   ```bash
   sudo target/release/mini-docker \
     --rootfs "$ROOTFS" \
     --cpu 20 \
     /bin/sh -lc "yes > /dev/null"
   ```

2. In another terminal, inspect `cat /sys/fs/cgroup/mini-docker-*/cpu.stat` while
   it runs; throttled time should increase.

3. Confirm that removing the limit (no `--cpu`) allows full utilisation.

4. Edge cases:

   - Specify `--cpu 0` to verify parsing errors.
   - Use `--cpu 100` with a multi-threaded load (e.g. `stress-ng --cpu 4`) to
     ensure quota equals the full period.

Mental model: imagine pouring sand through an hourglass—the quota constrains the
flow, and you should see grains (CPU cycles) waiting.

## Filesystem Isolation Checks

1. From inside the container, create `/tmp/marker` and verify it does not appear
   on the host outside the rootfs.
2. Attempt to access a host-only path such as `/etc/hosts`; it should reference
   the container's copy (or be absent) rather than the host file.
3. Mount persistence check: restart the container and ensure `/tmp` is empty
   because it is backed by a fresh tmpfs each run.
4. Edge cases:

   - Bind mount a host directory into the rootfs before launching and ensure the
     runtime honours the pre-created mountpoint.
   - Deliberately leave `/proc` missing to confirm the runtime creates it.

## Networking Containment Checks

1. Validate hostname isolation (`hostname` should show the configured value).
2. By default, the network namespace has no configured interfaces. Inside the
   container, run `ip link` to confirm only `lo` exists and is down.
3. Edge cases:

   - Attempt `ping 1.1.1.1` and expect failure due to missing network setup.
   - If you add a veth pair manually, ensure the runtime still functions, proving
     it does not assume specific interfaces.

Mental model: visualize pulling the Ethernet cable—unless you wire networking
back in, the container remains offline.

## Failure Mode Drills

Perform these to build confidence that error paths are graceful:

- **Cgroup unavailable**: Temporarily remount a tmpfs over `/sys/fs/cgroup` (in a
  test VM) and verify the runtime errors with `cgroup v2 is required`.
- **Mount failure**: Make `/proc` inside the rootfs read-only and check that the
  runtime surfaces the mount error.
- **Child panic**: Run a non-existent binary; ensure the error prints the failing
  `execvp` message.
- **Signal propagation**: Send `SIGTERM` to the parent process and watch the
  container terminate via `waitpid`.

## Rust Unit Tests

Automated tests focus on pure-Rust helpers that do not touch privileged
interfaces:

```bash
cargo test
```

The suite currently validates:

- `parse_memory` handles decimal values, suffixes (`k`, `m`, `g`), and rejects
  malformed input.
- `parse_cpu_percent` enforces the `1..=100` range and surfaces friendly errors.
- `unique_suffix` produces monotonically non-zero identifiers under normal clock
  conditions.

Extend the tests as you refactor: use the "invariant guardian" mental model—each
unit test guards a property that should never regress.

## Troubleshooting Checklist

When something misbehaves, walk this decision tree:

1. **Binary fails to start** → Rebuild with `cargo clean && cargo build` to rule
   out stale artefacts.
2. **Permission denied** → Confirm you are root and SELinux/AppArmor is not
   blocking mounts (inspect `dmesg`).
3. **cgroup errors** → Check that `systemd.unified_cgroup_hierarchy=1` is enabled
   and no legacy controllers interfere.
4. **Mount issues** → Ensure the rootfs directories exist and are writable before
   launching.
5. **Networking surprises** → If you expected connectivity, configure veth pairs
   manually; the runtime intentionally leaves the namespace bare.

## Further Exploration

- Add CLI flags for bind mounts or environment variables.
- Integrate seccomp filters to restrict syscalls.
- Experiment with user namespaces to drop root inside the container.
- Port the runtime to other languages for cross-paradigm comparison.

This documentation should equip you to reason about the runtime like a systems
engineer: hypothesise, test, observe, and iterate.
