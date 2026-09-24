#!/usr/bin/env python3
"""Run curated PPU checkin tests and produce a merged JUnit XML report.

This script drives the 80 curated PPU integration tests used by the
triton-for-sail CI pipeline.  It executes each test via pytest, collects
per-test JUnit XML fragments, merges them into a single report, and
optionally dumps environment metadata for downstream reporting.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class TestConfig:
    """单个测试的配置信息"""
    file_path: str                       # 测试文件路径
    test_filter: Optional[str] = None    # pytest -k 过滤或 ::node_id
    extra_args: List[str] = field(default_factory=list)  # 额外 pytest 参数
    skip_boards: List[str] = field(default_factory=list)  # 在指定板卡上跳过 (e.g. ["OAM-810E"])

    @property
    def display_name(self) -> str:
        """用于日志输出的可读名称"""
        name = self.file_path
        if self.test_filter:
            name = f"{name}::{self.test_filter}"
        return name


@dataclass
class TestResult:
    """单个测试运行的结果"""
    config: TestConfig
    returncode: int = -1
    duration: float = 0.0
    xml_path: Optional[str] = None
    stdout: str = ""
    stderr: str = ""

    @property
    def passed(self) -> bool:
        return self.returncode == 0

    @property
    def status_label(self) -> str:
        return "PASS" if self.passed else "FAIL"


# ---------------------------------------------------------------------------
# 默认测试配置
# ---------------------------------------------------------------------------

def get_default_test_configs(test_dir: str) -> List[TestConfig]:
    unit = os.path.join(test_dir, "python", "test", "unit")

    # 主跑批的 --ignore 列表 (绝对路径, 对应 Makefile test-unit 的排除项)
    main_ignored = [
        "language/test_line_info.py",
        "language/test_subprocess.py",
        "test_debug.py",
        "plugins/test_dialect_plugin.py",
        "plugins/test_plugin.py",
        "ppu/perf",
        "ppu/aiu/test_aiu_binary.py",
        "ppu/aiu/test_aiu_chained_dot.py",
        "ppu/aiu/test_aiu_load_padding.py",
        "ppu/aiu/test_aiu_block_ptr_load.py",
    ]
    main_args = ["--tb=short", "-n", "8"]
    for rel in main_ignored:
        main_args.append(f"--ignore={os.path.join(unit, rel)}")

    return [
        # ------------------------ test-unit ----------------------
        # 1) 主跑批: 整个 python/test/unit，但是ignore一部分文件
        TestConfig(
            file_path=unit,
            extra_args=main_args,
        ),
        # 2) subprocess 测试会 spawn 子进程, 单独跑
        TestConfig(
            file_path=os.path.join(unit, "language", "test_subprocess.py"),
            extra_args=["--tb=short", "-n", "8"],
        ),
        # 3) test_debug 需要进程隔离 (--forked)
        TestConfig(
            file_path=os.path.join(unit, "test_debug.py"),
            extra_args=["--tb=short", "-n", "8", "--forked"],
        ),
        # 4) line info 测试
        TestConfig(
            file_path=os.path.join(unit, "language", "test_line_info.py"),
            extra_args=["--tb=short"],
        ),
        # ------------------------ test-regression ----------------------
        # 5) regression 回归测试
        TestConfig(
            file_path=os.path.join(test_dir, "python", "test", "regression"),
            extra_args=["--tb=short", "-s", "-n", "8"],
        ),
        # ------------------------ test-gsan ----------------------
        # gsan 测试套件 (triton-for-sail 暂无 python/test/gsan 目录, 先注释)
        # TestConfig(
        #     file_path=os.path.join(test_dir, "python", "test", "gsan"),
        #     extra_args=["--tb=short", "-s", "-m", "xdist_group"],
        # ),
        # ------------------------ test-gluon ----------------------
        # 7) gluon 教程
        TestConfig(
            file_path=os.path.join(test_dir, "python", "tutorials", "gluon"),
            extra_args=["--tb=short", "-v"],
        ),
        # 8) gluon 前端测试套件
        TestConfig(
            file_path=os.path.join(test_dir, "python", "test", "gluon"),
            extra_args=["--tb=short", "-n", "1"],
        ),
        # ------------------------ test-triton-kernels ----------------------
        # 9) triton_kernels 套件
        TestConfig(
            file_path=os.path.join(test_dir, "python", "triton_kernels", "tests"),
            extra_args=["--tb=short", "-n", "6"],
        ),
        # ------------------------ test-proton ----------------------
        # 10) proton 全部测试
        TestConfig(
            file_path=os.path.join(test_dir, "third_party", "proton", "test"),
            extra_args=["--tb=short", "-s", "-n", "8"],
        ),
    ]


# ---------------------------------------------------------------------------
# 测试执行
# ---------------------------------------------------------------------------

def run_single_test(
    config: TestConfig,
    index: int,
    verbose: bool = False,
) -> TestResult:
    """
    运行单个 pytest 测试，生成临时 JUnit XML 结果文件。

    参数:
        config:  测试配置
        index:   测试序号（用于生成唯一临时文件名）
        verbose: 是否输出详细信息

    返回:
        TestResult 包含执行结果、计时和 XML 路径
    """
    result = TestResult(config=config)
    temp_xml = f"results_{index}.xml"

    # 构建 pytest 命令
    target = config.file_path
    if config.test_filter:
        target = f"{target}::{config.test_filter}"

    cmd: List[str] = [
        "pytest", "-v", "-s",
        target,
        f"--junitxml={temp_xml}",
    ]
    # 追加额外参数（如 -x, --timeout 等）
    if config.extra_args:
        cmd.extend(config.extra_args)

    # 打印运行信息
    print(f"\n{'='*70}")
    print(f"[{index + 1}] 正在运行: {config.display_name}")
    print(f"    命令: {' '.join(cmd)}")
    if verbose:
        print(f"    文件路径: {config.file_path} | 存在: {os.path.exists(config.file_path)}")
    print(f"{'='*70}")

    start_time = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=not verbose,  # verbose 模式下直接输出到终端
            text=True,
        )
        result.returncode = proc.returncode
        if not verbose:
            result.stdout = proc.stdout or ""
            result.stderr = proc.stderr or ""

        if proc.returncode == 0:
            print(f"  ✅ 测试通过: {config.display_name}")
        else:
            print(f"  ❌ 测试失败 (返回码={proc.returncode}): {config.display_name}")
            # 非 verbose 模式下，失败时打印 stderr 帮助调试
            if not verbose and result.stderr:
                print(f"  --- stderr 输出 (最后 20 行) ---")
                for line in result.stderr.strip().splitlines()[-20:]:
                    print(f"    {line}")

    except FileNotFoundError:
        print(f"  ❌ 找不到 pytest 命令，请确认已安装 pytest")
        result.returncode = -1
    except Exception as e:
        print(f"  ❌ 执行异常: {e}")
        result.returncode = -1
    finally:
        # 无论成功或失败，都记录生成的 XML 文件
        result.duration = time.time() - start_time
        if os.path.exists(temp_xml):
            result.xml_path = temp_xml
            if verbose:
                print(f"  📄 已生成临时 XML: {temp_xml}")
        else:
            if verbose:
                print(f"  ⚠️ 未生成 XML 文件: {temp_xml}")

    return result


# ---------------------------------------------------------------------------
# JUnit XML 合并
# ---------------------------------------------------------------------------

def merge_junit_xml(
    xml_files: List[str],
    output_path: str,
    verbose: bool = False,
) -> bool:
    """
    将多个 JUnit XML 结果文件合并为一个。

    支持两种常见 pytest 输出结构:
      - <testsuites><testsuite>...</testsuite></testsuites>
      - <testsuite>...</testsuite>  (直接作为根元素)

    参数:
        xml_files:   待合并的 XML 文件路径列表
        output_path: 合并后的输出文件路径
        verbose:     是否输出详细信息

    返回:
        合并是否成功（True/False）
    """
    # 初始化聚合统计
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    total_time = 0.0
    all_testcases: List[ET.Element] = []

    if not xml_files:
        print("⚠️ 没有 XML 文件需要合并，将生成空的结果文件")
    else:
        for xml_file in xml_files:
            # 检查文件是否存在且非空
            if not os.path.exists(xml_file):
                print(f"  ⚠️ 跳过不存在的文件: {xml_file}")
                continue
            if os.path.getsize(xml_file) == 0:
                print(f"  ⚠️ 跳过空文件: {xml_file}")
                continue

            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
            except ET.ParseError as e:
                print(f"  ⚠️ 解析 {xml_file} 失败 (XML 格式错误): {e}")
                continue
            except Exception as e:
                print(f"  ⚠️ 读取 {xml_file} 失败: {e}")
                continue

            # 收集所有 <testsuite> 元素
            testsuites: List[ET.Element] = []
            if root.tag == "testsuites":
                # 结构: <testsuites><testsuite>...</testsuite></testsuites>
                testsuites = root.findall("testsuite")
            elif root.tag == "testsuite":
                # 结构: <testsuite>...</testsuite> 直接作为根
                testsuites = [root]
            else:
                print(f"  ⚠️ {xml_file} 根元素非 testsuites/testsuite，跳过")
                continue

            file_test_count = 0
            for ts in testsuites:
                # 提取并聚合统计
                total_tests += int(ts.get("tests", "0"))
                total_failures += int(ts.get("failures", "0"))
                total_errors += int(ts.get("errors", "0"))
                total_skipped += int(ts.get("skipped", "0"))
                try:
                    total_time += float(ts.get("time", "0.0"))
                except ValueError:
                    pass

                # 提取所有 testcase 元素
                for tc in ts.findall("testcase"):
                    all_testcases.append(tc)
                    file_test_count += 1

            if verbose:
                print(f"  📋 已合并 {xml_file}: {file_test_count} 个测试用例")

    # 构建最终的 XML 结构
    final_root = ET.Element("testsuites")
    merged_suite = ET.SubElement(
        final_root,
        "testsuite",
        name="pytest-merged",
        tests=str(total_tests),
        failures=str(total_failures),
        errors=str(total_errors),
        skipped=str(total_skipped),
        time=f"{total_time:.3f}",
    )
    for tc in all_testcases:
        merged_suite.append(tc)

    # 写入最终 XML
    try:
        final_tree = ET.ElementTree(final_root)
        ET.indent(final_tree, space="  ")  # Python 3.9+ 格式化输出
    except AttributeError:
        # Python < 3.9 没有 ET.indent，跳过格式化
        final_tree = ET.ElementTree(final_root)

    final_tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"\n✅ 测试结果已合并到: {output_path}")
    print(f"   测试总数={total_tests}, 失败={total_failures}, "
          f"错误={total_errors}, 跳过={total_skipped}, "
          f"耗时={total_time:.3f}s")
    return True


# ---------------------------------------------------------------------------
# 清理临时文件
# ---------------------------------------------------------------------------

def cleanup_temp_files(xml_files: List[str], verbose: bool = False) -> None:
    """
    清理临时生成的 XML 文件。

    参数:
        xml_files: 待清理的文件路径列表
        verbose:   是否输出详细信息
    """
    if not xml_files:
        return

    cleaned = 0
    for f in xml_files:
        try:
            if os.path.exists(f):
                os.remove(f)
                cleaned += 1
                if verbose:
                    print(f"  🗑️ 已删除: {f}")
        except OSError as e:
            print(f"  ⚠️ 删除 {f} 失败: {e}")

    print(f"🧹 清理完成: 已删除 {cleaned}/{len(xml_files)} 个临时文件")


# ---------------------------------------------------------------------------
# 失败测试的合成 XML
# ---------------------------------------------------------------------------

def _create_synthetic_failure_xml(result: TestResult, output_path: str) -> None:
    """Create a minimal JUnit XML for a test that failed without producing XML.

    When pytest crashes during collection or early setup, it may not write any
    JUnit XML.  This function synthesises a one-testcase XML containing the
    failure information so that the merged report still shows the failure.
    """
    root = ET.Element(
        "testsuite",
        name="synthetic",
        tests="1",
        failures="1",
        errors="0",
        skipped="0",
        time=f"{result.duration:.3f}",
    )
    test_name = result.config.test_filter or os.path.basename(result.config.file_path)
    tc = ET.SubElement(
        root, "testcase",
        classname=result.config.file_path,
        name=test_name,
        time=f"{result.duration:.3f}",
    )
    fail_elem = ET.SubElement(
        tc, "failure",
        message=f"pytest exited with code {result.returncode} (no XML produced)",
    )
    details = result.stderr or result.stdout or ""
    if details:
        fail_elem.text = details[-2000:]
    else:
        fail_elem.text = (
            f"Test process exited with return code {result.returncode} "
            f"but did not produce JUnit XML output. "
            f"This usually means pytest crashed during collection or early setup."
        )
    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


# ---------------------------------------------------------------------------
# 摘要报告
# ---------------------------------------------------------------------------

def print_summary(results: List[TestResult], output_xml: str) -> None:
    """
    打印测试执行的摘要表格。

    参数:
        results:    所有测试运行结果
        output_xml: 合并后的 XML 文件路径
    """
    print(f"\n{'='*70}")
    print("                          测试执行摘要")
    print(f"{'='*70}")

    if not results:
        print("  (无测试运行)")
    else:
        # 表头
        print(f"  {'序号':<6}{'状态':<8}{'耗时':>10}  {'测试用例'}")
        print(f"  {'-'*6}{'-'*8}{'-'*10}  {'-'*40}")

        passed_count = 0
        failed_count = 0
        total_duration = 0.0

        for i, r in enumerate(results, start=1):
            status = r.status_label
            duration_str = f"{r.duration:.2f}s"
            name = r.config.display_name
            # 截断过长的名称
            if len(name) > 60:
                name = "..." + name[-57:]

            marker = "✅" if r.passed else "❌"
            print(f"  {i:<6}{marker} {status:<5}{duration_str:>10}  {name}")

            if r.passed:
                passed_count += 1
            else:
                failed_count += 1
            total_duration += r.duration

        print(f"  {'-'*6}{'-'*8}{'-'*10}  {'-'*40}")
        print(f"  {'合计':<6}{'':8}{total_duration:>9.2f}s  "
              f"通过={passed_count}, 失败={failed_count}, "
              f"总计={len(results)}")

    print(f"\n  📄 合并结果文件: {os.path.abspath(output_xml)}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# 环境信息收集
# ---------------------------------------------------------------------------

def collect_env_info() -> dict:
    """
    收集当前运行环境的版本信息（Python、PyTorch、Triton、HGCC、PPU SDK 等）。

    返回:
        包含环境信息的字典
    """
    info = {}
    # Python version
    info["Python"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    # PyTorch
    try:
        import torch
        info["PyTorch"] = torch.__version__
    except ImportError:
        pass
    # Triton
    try:
        import triton
        info["Triton"] = triton.__version__
    except ImportError:
        pass
    # HGCC compiler
    try:
        r = subprocess.run(["hgcc", "--version"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            output = r.stdout.strip()
            # Search ALL lines for version number (may not be on first line)
            # Patterns: "version X.Y.Z", "hgcc X.Y.Z", or standalone semver
            import re
            m = re.search(r"version\s+([\d][\w.\-]+)", output)
            if not m:
                m = re.search(r"hgcc\s+([\d]+\.[\d]+[\w.\-]*)", output)
            if not m:
                m = re.search(r"(\d+\.\d+\.\d+[\w.\-]*)", output)
            info["HGCC"] = m.group(1) if m else output.split("\n")[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # PPU SDK
    try:
        sdk_path = os.environ.get("PPU_SDK", "")
        if sdk_path:
            info["PPU SDK"] = sdk_path
            # Read release.yaml for the actual SDK version/build info.
            # Parse all top-level `key: value` pairs generically (no yaml dependency)
            # so we capture whatever fields the SDK provides.
            release_yaml = os.path.join(sdk_path, "release.yaml")
            if os.path.isfile(release_yaml):
                import re
                with open(release_yaml) as f:
                    for line in f:
                        line = line.rstrip("\n")
                        # skip comments, blank lines, list items and nested keys
                        if not line.strip() or line.lstrip().startswith("#"):
                            continue
                        if line[0] in (" ", "\t", "-"):
                            continue
                        m = re.match(r"^([\w.\-]+)\s*:\s*(.+?)\s*$", line)
                        if m:
                            k = m.group(1).strip()
                            v = m.group(2).strip().strip('"').strip("'")
                            if v:
                                info[f"PPU SDK {k}"] = v
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """
    主入口：解析命令行参数，运行测试，合并结果，输出摘要。

    参数:
        argv: 命令行参数列表（默认使用 sys.argv）

    返回:
        退出码（0=全部通过，1=有失败）
    """
    parser = argparse.ArgumentParser(
        description="通用 pytest 测试运行与 JUnit XML 合并框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例用法:\n"
            "  python run_checkin_tests.py\n"
            "  python run_checkin_tests.py -o result.xml --test-dir /path/to/repo\n"
            "  python run_checkin_tests.py --keep-temp -v\n"
        ),
    )
    parser.add_argument(
        "--output", "-o",
        default="test-results.xml",
        help="合并后的 XML 输出文件路径 (默认: test-results.xml)",
    )
    parser.add_argument(
        "--test-dir",
        default=".",
        help="测试文件的基础目录 (默认: 当前目录)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="保留临时 XML 文件，不在合并后删除",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="输出详细信息",
    )
    parser.add_argument(
        "--env-output",
        default="",
        help="收集环境信息并写入指定的 JSON 文件路径（留空则跳过）",
    )

    args = parser.parse_args(argv)

    # ---- 加载测试配置 ----
    # 使用默认配置；如需自定义，可修改此处或通过外部配置文件加载
    test_configs: List[TestConfig] = get_default_test_configs(args.test_dir)

    if not test_configs:
        print("⚠️ 没有配置任何测试用例")

    print(f"📋 共 {len(test_configs)} 个测试组待运行")
    print(f"📂 测试基础目录: {os.path.abspath(args.test_dir)}")
    print(f"📄 输出文件: {args.output}")

    # ---- 逐个运行测试 ----
    current_board = os.environ.get("BOARD_TYPE", "")
    results: List[TestResult] = []
    skipped_by_board = 0
    for idx, cfg in enumerate(test_configs):
        if cfg.skip_boards and current_board in cfg.skip_boards:
            print(f"\n⏭️ [{idx+1}/{len(test_configs)}] {cfg.display_name} — skipped on {current_board}")
            skipped_by_board += 1
            continue
        result = run_single_test(cfg, idx, verbose=args.verbose)
        results.append(result)
    if skipped_by_board:
        print(f"\n⏭️ {skipped_by_board} test(s) skipped for board {current_board}")

    # ---- 为未生成 XML 的失败测试创建合成 XML ----
    for idx, r in enumerate(results):
        if not r.passed and r.xml_path is None:
            synthetic_path = f"results_{idx}_synthetic.xml"
            _create_synthetic_failure_xml(r, synthetic_path)
            r.xml_path = synthetic_path
            print(f"  ⚠️ Created synthetic failure XML for: {r.config.display_name}")

    # ---- 收集所有生成的 XML 文件 ----
    xml_files: List[str] = [
        r.xml_path for r in results if r.xml_path is not None
    ]

    # ---- 合并 XML ----
    print(f"\n{'─'*70}")
    print("正在合并 JUnit XML 结果...")
    merge_junit_xml(xml_files, args.output, verbose=args.verbose)

    # ---- 清理临时文件 ----
    if not args.keep_temp:
        cleanup_temp_files(xml_files, verbose=args.verbose)
    else:
        print(f"ℹ️ 保留临时文件 (--keep-temp): {xml_files}")

    # ---- 打印摘要 ----
    print_summary(results, args.output)

    # ---- 收集环境信息（可选） ----
    if args.env_output:
        env_info = collect_env_info()
        with open(args.env_output, "w") as f:
            json.dump(env_info, f, indent=2)
        print(f"📋 环境信息已写入: {args.env_output}")
        for k, v in env_info.items():
            print(f"  {k}: {v}")

    # ---- 返回退出码 ----
    has_failures = any(not r.passed for r in results)
    return 1 if has_failures else 0


if __name__ == "__main__":
    sys.exit(main())
