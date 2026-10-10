#!/usr/bin/env python3
"""Run curated PPU checkin tests and produce a merged JUnit XML report.

This script drives the PPU integration tests used by the triton-for-sail CI
pipeline.  It executes each test group via pytest, collects per-group JUnit
XML fragments, merges them into a single report, and optionally dumps
environment metadata for downstream reporting.

Adapted for release/3.6.x test tree (directory-level TestConfig with xdist,
300s timeout, environment variable support, and collection-failure synthetic
JUnit).
"""

import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional


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
    env: Dict[str, str] = field(default_factory=dict)     # 注入给 pytest 子进程的环境变量

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
# 默认测试配置 — curated ~80 checkin tests for release/3.6.x
# ---------------------------------------------------------------------------
# 从全目录收集收敛为约 80 条精选用例, CI 预算约两小时.
# 覆盖: 核心语言语义 / runtime / tools / PPU AIU / fused attention / MXFP / FLA.
# 并行度 ≤ 4 (主批次 -n 4); MAX_JOBS=16 保留在 workflow 层.
# 仅修改 CI harness, 不修改任何测试源码.
# ---------------------------------------------------------------------------

def get_default_test_configs(test_dir: str) -> List[TestConfig]:
    """Return curated checkin test list for release/3.6.x PPU CI."""
    unit = os.path.join(test_dir, "python", "test", "unit")
    lang = os.path.join(unit, "language")
    runtime = os.path.join(unit, "runtime")
    tools = os.path.join(unit, "tools")
    aiu = os.path.join(unit, "ppu", "aiu")
    perf = os.path.join(unit, "ppu", "perf")
    models = os.path.join(unit, "ppu", "models")
    mxfp = os.path.join(unit, "ppu", "mxfp")

    return [
        # ==============================================================
        # 语言层 - frontend / TTIR / TTGIR:
        # block pointer / 算术 / 位运算 / 比较 / broadcast / slice 错误 /
        # reduce / where / random / print
        # ==============================================================
        TestConfig(
            # parametrize index 216 = (float32,float32), n=1024, padding=None, boundary=None
            file_path=os.path.join(lang, "test_block_pointer.py"),
            test_filter="test_block_copy[dtypes_str216-1024-None-None]",
        ),
        TestConfig(
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_bin_op[1-int32-int8-+]",
        ),
        TestConfig(
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_floordiv[1-uint8-uint32]",
        ),
        TestConfig(
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_bitwise_op[1-int8-int8-&0]",
        ),
        TestConfig(
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_compare_op[1-int8-int8-==-real-real]",
        ),
        TestConfig(
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_broadcast[float64]",
        ),
        TestConfig(
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_invalid_slice",
        ),
        TestConfig(
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_reduce1d[1-min-int8-32]",
        ),
        TestConfig(
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_value_specialization_overflow[-9223372036854775808-False]",
        ),
        TestConfig(
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_where[1-bfloat16]",
        ),
        TestConfig(
            file_path=os.path.join(lang, "test_random.py"),
            test_filter="test_randint[10-0-int32-True]",
        ),
        TestConfig(
            file_path=os.path.join(lang, "test_subprocess.py"),
            test_filter="test_print[device_print-int8]",
        ),

        # ==============================================================
        # 语言层 - 类型转换: FpToFp 上/下转换 (fp8 双向), FpToInt narrow,
        # identity cast
        # ==============================================================
        TestConfig(
            # bf16 -> fp8_e5m2 (FpToFp downcast 到 fp8)
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_cast[1-bfloat16-float8_e5m2-False-32]",
        ),
        TestConfig(
            # fp8_e5m2 -> bf16 (fp8 -> 高精度浮点 upcast)
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_cast[1-float8_e5m2-bfloat16-False-1024]",
        ),
        TestConfig(
            # fp64 -> uint8 (FpToInt, narrowing to unsigned)
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_cast[1-float64-uint8-False-1024]",
        ),
        TestConfig(
            # int8 -> int8 (identity cast)
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_cast[1-int8-int8-False-1024]",
        ),

        # ==============================================================
        # 语言层 - reduce / scan / sort / flip
        # ==============================================================
        TestConfig(
            # 多维 reduce + permute 串联
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_chained_reductions[in_shape0-perm0-red_dims0]",
        ),
        TestConfig(
            # 2D reduce min, shape=(2,32) float32 axis=0
            # NOTE: shape index depends on parametrize ordering;
            # shape60=(2,32) for configs2 first entry on 3.6.x (60 configs1 entries precede)
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_reduce[1-min-float32-shape60-0-False]",
        ),
        TestConfig(
            # tl.cumsum 1D scan
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_scan_1d[8-8]",
        ),
        TestConfig(
            # tl.sort: bitonic sort
            file_path=os.path.join(lang, "test_standard.py"),
            test_filter="test_sort[int32-False-None-1-1]",
        ),
        TestConfig(
            # tl.flip
            file_path=os.path.join(lang, "test_standard.py"),
            test_filter="test_flip[0-int32-1-16-64]",
        ),

        # ==============================================================
        # 语言层 - transpose / permute / histogram
        # ==============================================================
        TestConfig(
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_transpose[int32]",
        ),
        TestConfig(
            # tl.permute (1,0), fp16 64x64
            # 3.6.x global index 2: shape2=(64,64), perm2=(1,0) (float8e4b15 takes 0-1)
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_permute[1-float16-shape2-perm2]",
        ),
        TestConfig(
            # tl.histogram
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_histogram[8-2]",
        ),

        # ==============================================================
        # 语言层 - 通用 tl.dot (非 AIU 加速)
        # ==============================================================
        TestConfig(
            # 最小尺寸 (16x16x16) fp16->fp32 dot
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_dot[1-16-16-16-4-False-False-none-ieee-float16-float32-1-None]",
        ),

        # ==============================================================
        # 语言层 - 数学 intrinsics (libdevice)
        # ==============================================================
        TestConfig(
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_math_op[float32-exp-x]",
        ),

        # ==============================================================
        # 语言层 - tl.range / pipelining
        # ==============================================================
        TestConfig(
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_tl_range_num_stages",
        ),
        TestConfig(
            file_path=os.path.join(lang, "test_pipeliner.py"),
            test_filter="test_pipeline_vecadd",
        ),
        TestConfig(
            file_path=os.path.join(lang, "test_pipeliner.py"),
            test_filter="test_pipeline_matmul[False]",
        ),
        TestConfig(
            file_path=os.path.join(lang, "test_pipeliner.py"),
            test_filter="test_pipeline_epilogue[1-0]",
        ),

        # ==============================================================
        # 语言层 - 原子操作
        # ==============================================================
        TestConfig(
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_atomic_cas[int32-1-None]",
        ),
        TestConfig(
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_tensor_atomic_rmw_block[1]",
        ),

        # ==============================================================
        # 语言层 - 控制流: if / while / for
        # ==============================================================
        TestConfig(
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_if_else",
        ),
        TestConfig(
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_while",
        ),
        TestConfig(
            # 反向 for-loop (iv<0)
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_for_iv[22-18--1]",
        ),

        # ==============================================================
        # 语言层 - masked load + padding
        # ==============================================================
        TestConfig(
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_masked_load[1-float32-128-2-0]",
        ),

        # ==============================================================
        # 语言层 - shape ops: expand_dims
        # ==============================================================
        TestConfig(
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_expand_dims",
        ),

        # ==============================================================
        # 语言层 - noinline 函数调用
        # ==============================================================
        TestConfig(
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_noinline[call_graph]",
        ),

        # ==============================================================
        # 语言层 - 数据重组: cat / join / split
        # ==============================================================
        TestConfig(
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_cat[int8-4]",
        ),
        TestConfig(
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_join",
        ),
        TestConfig(
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_split",
        ),

        # ==============================================================
        # 语言层 - 3D dot / scaled dot
        # ==============================================================
        TestConfig(
            # 3D batched dot, B=1, int8
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_dot3d[1-1-64-64-64-32-32-int8-int8]",
        ),
        TestConfig(
            # tl.dot_scaled: MX 浮点缩放 matmul, e2m1 格式
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_scaled_dot[32-32-64-True-True-False-e2m1-e4m3-4-16-1]",
            skip_boards=["OAM-810E"],
        ),

        # ==============================================================
        # 语言层 - 构造 / 索引: full / arange / gather
        # ==============================================================
        TestConfig(
            # tl.full: shape=(128,) int32 — shape2 maps to (128,) on 3.6.x
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_full[shape2-int32]",
        ),
        TestConfig(
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_arange[1-0]",
        ),
        TestConfig(
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_gather",
        ),

        # ==============================================================
        # 语言层 - inline asm
        # ==============================================================
        TestConfig(
            file_path=os.path.join(lang, "test_core.py"),
            test_filter="test_inline_asm[1]",
        ),

        # ==============================================================
        # 语言层 - 类型转换 (test_conversions): fp16->fp32 upcast
        # ==============================================================
        TestConfig(
            file_path=os.path.join(lang, "test_conversions.py"),
            test_filter="test_typeconvert_upcast[float16-float32]",
        ),

        # ==============================================================
        # 语言层 - frontend AST
        # ==============================================================
        TestConfig(
            file_path=os.path.join(lang, "test_frontend.py"),
            test_filter="test_assign_attribute",
        ),
        TestConfig(
            file_path=os.path.join(lang, "test_frontend.py"),
            test_filter="test_constexpr_function_from_jit",
        ),

        # ==============================================================
        # 语言层 - tuple
        # ==============================================================
        TestConfig(
            file_path=os.path.join(lang, "test_tuple.py"),
            test_filter="test_index[0]",
        ),

        # ==============================================================
        # Runtime 层: cache / autotuner / launch / driver / build / subproc
        # ==============================================================
        TestConfig(
            file_path=os.path.join(runtime, "test_cache.py"),
            test_filter="test_reuse",
        ),
        TestConfig(
            file_path=os.path.join(runtime, "test_autotuner.py"),
            test_filter="test_kwargs[False]",
        ),
        TestConfig(
            file_path=os.path.join(runtime, "test_launch.py"),
            test_filter="test_metadata",
        ),
        TestConfig(
            file_path=os.path.join(runtime, "test_cache.py"),
            test_filter="test_nochange",
        ),
        TestConfig(
            file_path=os.path.join(runtime, "test_driver.py"),
            test_filter="test_is_lazy",
        ),
        TestConfig(
            file_path=os.path.join(runtime, "test_build.py"),
            test_filter="test_compile_module",
        ),
        TestConfig(
            file_path=os.path.join(runtime, "test_subproc.py"),
            test_filter="test_compile_in_subproc",
        ),

        # ==============================================================
        # 基础设施 / 工具: linear layout / filecheck / irsource
        # ==============================================================
        TestConfig(
            file_path=os.path.join(tools, "test_linear_layout.py"),
            test_filter="test_compose",
        ),
        TestConfig(
            file_path=os.path.join(unit, "test_filecheck.py"),
            test_filter="test_filecheck_positive",
        ),
        TestConfig(
            file_path=os.path.join(tools, "test_linear_layout.py"),
            test_filter="test_invert",
        ),
        TestConfig(
            file_path=os.path.join(tools, "test_irsource.py"),
            test_filter="test_mlir_attribute_parsing",
        ),

        # ==============================================================
        # 基础设施 - knobs / static_assert
        # ==============================================================
        TestConfig(
            file_path=os.path.join(unit, "test_knobs.py"),
            test_filter="test_knobs_utils",
        ),
        TestConfig(
            file_path=os.path.join(unit, "test_debug.py"),
            test_filter="test_static_assert[True]",
        ),

        # ==============================================================
        # PPU AIU - load: 普通 2D load, block-pointer load
        # ==============================================================
        TestConfig(
            file_path=os.path.join(aiu, "test_aiu_load.py"),
            test_filter="test_aiu_load[64-64-1024-1024-2-2]",
        ),
        TestConfig(
            file_path=os.path.join(aiu, "test_aiu_tensor_ptr.py"),
            test_filter="test_aiu_load[64-64-1024-1024-2-2]",
        ),

        # ==============================================================
        # PPU AIU - dot (主路径)
        # ==============================================================
        TestConfig(
            file_path=os.path.join(aiu, "test_aiu_dot.py"),
            test_filter="test_aiu_matmul[False-64-64-64-1024-1024-1024-2-2]",
        ),

        # ==============================================================
        # PPU 端到端 - fused attention
        # ==============================================================
        TestConfig(
            # 通用 (非 AIU) flash-attention
            file_path=os.path.join(perf, "06-fused-attention.py"),
            test_filter="test_op[True-1-2-1024-64]",
        ),
        TestConfig(
            # AIU fp16 flash-attention
            file_path=os.path.join(perf, "06-fused-attention-aiu.py"),
            test_filter="test_op_fp16[True-1-2-1024-64]",
        ),

        # ==============================================================
        # PPU AIU - addmm: bias + matmul 融合
        # ==============================================================
        TestConfig(
            file_path=os.path.join(aiu, "test_aiu_addmm.py"),
            test_filter="test_aiu_addmm[0.001-32-32-32-1024-1024-1024-1-2]",
        ),

        # ==============================================================
        # PPU AIU - binary 逐元素运算
        # ==============================================================
        TestConfig(
            file_path=os.path.join(aiu, "test_aiu_binary.py"),
            test_filter="test_aiu_binary[False-32-32-1024-1024-1-2-+]",
        ),
        TestConfig(
            file_path=os.path.join(aiu, "test_aiu_binary.py"),
            test_filter="test_aiu_binary[False-32-32-1024-1024-1-2-*]",
        ),

        # ==============================================================
        # PPU AIU - dot fp8 / dot order
        # ==============================================================
        TestConfig(
            file_path=os.path.join(aiu, "test_aiu_dot_fp8_order.py"),
            test_filter="test_aiu_matmul_fp8_with_order[32-32-32-512-512-512-2-2]",
            skip_boards=["OAM-810E"],
        ),
        TestConfig(
            file_path=os.path.join(aiu, "test_aiu_dot_order.py"),
            test_filter="test_aiu_matmul[32-32-32-1024-1024-1024-1-2]",
        ),

        # ==============================================================
        # PPU AIU - tensor_ptr order
        # ==============================================================
        TestConfig(
            file_path=os.path.join(aiu, "test_aiu_tensor_ptr_order.py"),
            test_filter="test_aiu_matmul[32-32-32-1024-1024-1024-1-2]",
        ),

        # ==============================================================
        # PPU AIU - dot 扩展: mixed_load + small block
        # ==============================================================
        TestConfig(
            file_path=os.path.join(aiu, "test_aiu_dot.py"),
            test_filter="test_aiu_matmul[True-32-32-32-1024-1024-1024-1-2]",
        ),
        TestConfig(
            file_path=os.path.join(aiu, "test_aiu_dot.py"),
            test_filter="test_aiu_matmul_small_block[False-16-16-16-128-128-128-1-2]",
        ),

        # ==============================================================
        # PPU models - FLA (Flash Linear Attention)
        # ==============================================================
        TestConfig(
            file_path=os.path.join(models, "test_fla_ops.py"),
            test_filter="test_chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
        ),

        # ==============================================================
        # PPU MXFP - blocked-scale MX 浮点 matmul
        # ==============================================================
        TestConfig(
            file_path=os.path.join(mxfp, "test_mxfp_matmul.py"),
            test_filter="test_blocked_scale_mxfp4[False-1-128-128-128-1024-512-256]",
            skip_boards=["OAM-810E"],
        ),

        # ==============================================================
        # PPU 端到端 - fused attention fp8 (AIU 加速)
        # ==============================================================
        TestConfig(
            file_path=os.path.join(perf, "06-fused-attention-aiu.py"),
            test_filter="test_op_fp8[True-1-2-1024-64]",
            skip_boards=["OAM-810E"],
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
        "pytest", "--tb=short", "--no-header", "--timeout=300",
        target,
        f"--junitxml={temp_xml}",
    ]
    # 追加额外参数（如 -x, --timeout 等）
    if config.extra_args:
        cmd.extend(config.extra_args)

    # 打印运行信息
    print(f"\n{'='*70}")
    print(f"[{index + 1}] 正在运行: {config.display_name}")
    print(f"    命令: pytest {target} + {len(cmd)-4} args")
    if config.env:
        print(f"    环境变量: {config.env}")
    if verbose:
        print(f"    文件路径: {config.file_path} | 存在: {os.path.exists(config.file_path)}", flush=True)
    print(f"{'='*70}", flush=True)

    start_time = time.time()
    run_env = os.environ.copy()
    if config.env:
        run_env.update(config.env)
    try:
        proc = subprocess.run(
            cmd,
            stdout=None,              # 始终流式输出到 CI log
            stderr=subprocess.PIPE,   # 捕获 stderr 用于失败诊断
            text=True,
            env=run_env,
        )
        result.returncode = proc.returncode
        result.stderr = proc.stderr or ""

        if proc.returncode == 0:
            print(f"  ✅ 测试通过: {config.display_name}", flush=True)
        else:
            print(f"  ❌ 测试失败 (返回码={proc.returncode}): {config.display_name}", flush=True)
            # 失败时打印 stderr 帮助调试（stdout 已流式输出，无需事后打印）
            if result.stderr:
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

    # ---- 返回退出码 ----
    has_failures = any(not r.passed for r in results)
    return 1 if has_failures else 0


if __name__ == "__main__":
    sys.exit(main())

