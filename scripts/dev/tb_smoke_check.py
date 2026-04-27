"""Fast standalone smoke test for the TB env's docker integration.

Exercises pull + run_detached + exec + cp + test.sh + remove_force on
ONE task, ONE rollout — without spinning up the full
orchestrator/trainer/inference stack. ~30 seconds end-to-end on a
warm registry. Catches docker-integration regressions ~80x faster
than a real sbatch.

Usage (on a compute node with docker access):
    python /tmp/tb_smoke_check.py [task_name]

Default task is `aimo-airline-departures`. Override via CLI to
test a specific task.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path("/home/seanbell/scalerl-prime-rl")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


async def main():
    from environments.terminal_bench.env import _DockerClient, load_environment

    task = sys.argv[1] if len(sys.argv) > 1 else "aimo-airline-departures"
    print(f"[smoke] testing task={task!r}")

    env = load_environment(
        tasks=[task],
        task_root="environments/terminal_bench/tasks",
        registry_host="rlgpu0:5000",
        build_local_images=True,
    )
    spec = env._task_specs[task]
    docker = env._docker

    # 1. _resolve_image (covers pull from registry)
    print(f"[smoke] _resolve_image({task})...")
    image = await env._resolve_image(spec)
    print(f"[smoke]   -> tag = {image!r}")
    assert await docker.image_exists(image), f"image_exists False after pull"
    print("[smoke]   image_exists OK")

    # 2. run_detached (covers semaphore + cgroup creation)
    container = f"tb-smoke-{task}-{__import__('uuid').uuid4().hex[:8]}"
    print(f"[smoke] run_detached({container})...")
    await docker.run_detached(
        image,
        name=container,
        labels={"prime_rl_tb_env": "1", "prime_rl_tb_run_id": "smoke", "prime_rl_tb_task": task},
    )
    print(f"[smoke]   container started")

    try:
        # 3. exec (covers bash -lc; ensure stdout/stderr capture works)
        print("[smoke] exec 'echo hello'...")
        rc, out, err = await docker.exec(container, "echo hello", timeout=10, working_dir="/")
        assert rc == 0, f"echo exit={rc} stderr={err!r}"
        assert out.strip() == "hello", f"unexpected stdout: {out!r}"
        print(f"[smoke]   exec OK (rc={rc}, out={out.strip()!r})")

        # 4. exec with WORKDIR override (covers /app pinning for model shell tool)
        print("[smoke] exec 'pwd' with working_dir=/app...")
        rc, out, err = await docker.exec(container, "pwd", timeout=10, working_dir="/app")
        assert rc == 0, f"pwd exit={rc} stderr={err!r}"
        assert out.strip() == "/app", f"unexpected pwd: {out!r}"
        print(f"[smoke]   working_dir override OK ({out.strip()!r})")

        # 5. mkdir + cp_into (covers the prep /tests path)
        print("[smoke] mkdir + cp_into /tests...")
        tests_dir = spec.dir / "tests"
        rc, out, err = await docker.exec(container, "mkdir -p /tests /logs/verifier", timeout=10, working_dir="/")
        assert rc == 0, f"mkdir exit={rc} stderr={err!r}"
        await docker.cp_into(f"{tests_dir}/.", container, "/tests")
        rc, out, err = await docker.exec(container, "ls /tests/test.sh", timeout=10, working_dir="/")
        assert rc == 0, f"test.sh missing after cp: stderr={err!r}"
        print("[smoke]   /tests prepared")

        # 6. bash test.sh (the critical reward path)
        print("[smoke] running test.sh (timeout=300)...")
        rc, out, err = await docker.exec(
            container,
            "cd /tests && bash test.sh",
            timeout=300,
            working_dir="/",
        )
        print(f"[smoke]   test.sh rc={rc} stdout_len={len(out)} stderr_len={len(err)}")
        if out:
            print(f"[smoke]   stdout tail: {out.strip()[-200:]!r}")
        if err:
            print(f"[smoke]   stderr tail: {err.strip()[-200:]!r}")

        # 7. read reward.txt
        print("[smoke] reading reward.txt...")
        rc, reward_txt, err = await docker.exec(
            container,
            "cat /logs/verifier/reward.txt 2>/dev/null || echo MISSING",
            timeout=10,
            working_dir="/",
        )
        reward_str = reward_txt.strip()
        if reward_str == "MISSING":
            print(f"[smoke]   ⚠ reward.txt MISSING after test.sh — this is the failure mode we've been chasing")
        else:
            print(f"[smoke]   reward.txt = {reward_str!r}")

    finally:
        # 8. remove_force
        print(f"[smoke] remove_force({container})...")
        try:
            await docker.remove_force(container)
            print("[smoke]   container removed")
        except Exception as e:
            print(f"[smoke]   remove_force failed: {e!r}")

    print("[smoke] DONE")


if __name__ == "__main__":
    asyncio.run(main())
