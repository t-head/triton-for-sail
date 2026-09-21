#!/usr/bin/env python3

import argparse
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
# 默认测试配置（保留原脚本的测试用例作为示例）
# ---------------------------------------------------------------------------

def get_default_test_configs(test_dir: str) -> List[TestConfig]:
    """
    返回默认的测试配置列表。
    保留原脚本中的测试用例配置，可按需修改或扩展。
    """
    return [
        # --------------------------------------------------------------
        # 语言层 - frontend / TTIR / TTGIR:
        # block pointer / 算术 / 位运算 / 比较 / broadcast / slice 错误 /
        # reduce / where / random / print
        # --------------------------------------------------------------
        TestConfig(
            file_path=os.path.join(test_dir, "python/test/unit/language/test_block_pointer.py"),
            test_filter="test_block_copy[dtypes_str36-1024-None-None]",
        ),
        TestConfig(
            file_path=os.path.join(test_dir, "python/test/unit/language/test_core.py"),
            test_filter="test_bin_op[1-int32-int8-+]",
        ),
        TestConfig(
            file_path=os.path.join(test_dir, "python/test/unit/language/test_core.py"),
            test_filter="test_floordiv[1-uint8-uint32]",
        ),
        TestConfig(
            file_path=os.path.join(test_dir, "python/test/unit/language/test_core.py"),
            test_filter="test_bitwise_op[1-int8-int8-&0]",
        ),
        TestConfig(
            file_path=os.path.join(test_dir, "python/test/unit/language/test_core.py"),
            test_filter="test_compare_op[1-int8-int8-==-real-real]",
        ),
        TestConfig(
            file_path=os.path.join(test_dir, "python/test/unit/language/test_core.py"),
            test_filter="test_broadcast[float64]",
        ),
        TestConfig(
            file_path=os.path.join(test_dir, "python/test/unit/language/test_core.py"),
            test_filter="test_invalid_slice",
        ),
        TestConfig(
            file_path=os.path.join(test_dir, "python/test/unit/language/test_core.py"),
            test_filter="test_reduce1d[1-min-int8-32]",
        ),
        TestConfig(
            file_path=os.path.join(test_dir, "python/test/unit/language/test_core.py"),
            test_filter="test_value_specialization_overflow[-9223372036854775808-False]",
        ),
        TestConfig(
            file_path=os.path.join(test_dir, "python/test/unit/language/test_core.py"),
            test_filter="test_where[1-bfloat16]",
        ),
        TestConfig(
            file_path=os.path.join(test_dir, "python/test/unit/language/test_random.py"),
            test_filter="test_randint[10-0-int32-True]",
        ),
        TestConfig(
            file_path=os.path.join(test_dir, "python/test/unit/language/test_subprocess.py"),
            test_filter="test_print[device_print-int8]",
        ),

        # --------------------------------------------------------------
        # 语言层 - 类型转换: 覆盖 FpToFp 上/下转换 (fp8 双向)、
        # FpToInt narrow (fp64->u8), 以及 identity cast (int8->int8)
        # --------------------------------------------------------------
        TestConfig(
            # bf16 -> fp8_e5m2 (FpToFp downcast 到 fp8)
            file_path=os.path.join(test_dir, "python/test/unit/language/test_core.py"),
            test_filter="test_cast[1-bfloat16-float8_e5m2-False-32]",
        ),
        TestConfig(
            # fp8_e5m2 -> bf16 (fp8 -> 高精度浮点 upcast)
            file_path=os.path.join(test_dir, "python/test/unit/language/test_core.py"),
            test_filter="test_cast[1-float8_e5m2-bfloat16-False-1024]",
        ),
        TestConfig(
            # fp64 -> uint8 (FpToInt, narrowing to unsigned)
            file_path=os.path.join(test_dir, "python/test/unit/language/test_core.py"),
            test_filter="test_cast[1-float64-uint8-False-1024]",
        ),
        TestConfig(
            # int8 -> int8 (同类型 cast, identity 路径)
            file_path=os.path.join(test_dir, "python/test/unit/language/test_core.py"),
            test_filter="test_cast[1-int8-int8-False-1024]",
        ),

        # --------------------------------------------------------------
        # 语言层 - reduce / scan / sort / flip: 覆盖 tl.standard 工具函数
        # 以及 ReduceOp / ScanOp 的 layout 处理
        # --------------------------------------------------------------
        TestConfig(
            # 多维 reduce + permute 串联, 压力测试 slice layout 在 reduce 链中的传播
            file_path=os.path.join(test_dir, "python/test/unit/language/test_core.py"),
            test_filter="test_chained_reductions[in_shape0-perm0-red_dims0]",
        ),
        TestConfig(
            # 2D reduce min, shape=(2,32) float32 axis=0
            # 验证 ReduceOp 2D + axis lowering (与 1D reduce 路径不同)
            # NOTE: shape60 为 reduce_configs1+configs2 拼接列表中的索引,
            # 上游若插入/重排 reduce_configs 会失效, 上线后请观察
            file_path=os.path.join(test_dir, "python/test/unit/language/test_core.py"),
            test_filter="test_reduce[1-min-float32-shape60-0-False]",
        ),
        TestConfig(
            # tl.cumsum 1D scan, 覆盖 ScanOp 主路径
            file_path=os.path.join(test_dir, "python/test/unit/language/test_core.py"),
            test_filter="test_scan_1d[8-8]",
        ),
        TestConfig(
            # tl.sort: bitonic sort lowering, 简单 1x1 case
            file_path=os.path.join(test_dir, "python/test/unit/language/test_standard.py"),
            test_filter="test_sort[int32-False-None-1-1]",
        ),
        TestConfig(
            # tl.flip: 反向排列, 验证 ReverseOp lowering
            file_path=os.path.join(test_dir, "python/test/unit/language/test_standard.py"),
            test_filter="test_flip[0-int32-1-16-64]",
        ),

        # --------------------------------------------------------------
        # 语言层 - transpose / permute / histogram:
        # 覆盖 TransOp / PermuteOp / HistogramOp lowering
        # --------------------------------------------------------------
        TestConfig(
            # 2D transpose (.T), 覆盖 TransOp + layout 转换
            file_path=os.path.join(test_dir, "python/test/unit/language/test_core.py"),
            test_filter="test_transpose[int32]",
        ),
        TestConfig(
            # tl.permute (1,0), fp16 64x64; 走 ConvertLayout 路径, 与 .T 不同入口
            # NOTE: shape2/perm2 为按 parametrize 行号生成的索引, 对应
            # (dtype=float16, shape=(64,64), perm=(1,0)) 这一组
            file_path=os.path.join(test_dir, "python/test/unit/language/test_core.py"),
            test_filter="test_permute[1-float16-shape2-perm2]",
        ),
        TestConfig(
            # tl.histogram: bin 计数, 覆盖 HistogramOp lowering
            file_path=os.path.join(test_dir, "python/test/unit/language/test_core.py"),
            test_filter="test_histogram[8-2]",
        ),

        # --------------------------------------------------------------
        # 语言层 - 通用 tl.dot (非 AIU 加速路径):
        # 走 DotOpToLLVM/DotOpConversion, 与 PPU AIU matmul 完全不同的 codegen
        # --------------------------------------------------------------
        TestConfig(
            # 最小尺寸 (16x16x16) fp16->fp32 dot, ieee precision, num_warps=4
            file_path=os.path.join(test_dir, "python/test/unit/language/test_core.py"),
            test_filter="test_dot[1-16-16-16-4-False-False-none-ieee-float16-float32-1-None]",
        ),

        # --------------------------------------------------------------
        # 语言层 - 数学 intrinsics (libdevice):
        # 覆盖 ConvertLibdeviceFuncToPPU 通路, 是任何数学算子改动的最小 smoke
        # --------------------------------------------------------------
        TestConfig(
            # tl.exp(x) on float32, 验证 libdevice -> PPU lowering 闭环
            file_path=os.path.join(test_dir, "python/test/unit/language/test_core.py"),
            test_filter="test_math_op[float32-exp-x]",
        ),

        # --------------------------------------------------------------
        # 语言层 - tl.range / pipelining: 覆盖软件流水线 (LoopPipeliner)
        # 的 num_stages / async_copy / vec-add / matmul / epilogue 路径
        # --------------------------------------------------------------
        TestConfig(
            # tl.range(num_stages=...) + matmul, 编译期检查 pipeline 是否生效
            file_path=os.path.join(test_dir, "python/test/unit/language/test_core.py"),
            test_filter="test_tl_range_num_stages",
        ),
        TestConfig(
            # 简单 vecadd, 检查 async_copy_global_to_local 数量与 NUM_STAGES 一致
            file_path=os.path.join(test_dir, "python/test/unit/language/test_pipeliner.py"),
            test_filter="test_pipeline_vecadd",
        ),
        TestConfig(
            # matmul 在 pipeliner 下的多 stage 调度 (非 scaled 路径)
            file_path=os.path.join(test_dir, "python/test/unit/language/test_pipeliner.py"),
            test_filter="test_pipeline_matmul[False]",
        ),
        TestConfig(
            # tl.range + epilogue, 覆盖 store 在 pipeline 之外的处理
            file_path=os.path.join(test_dir, "python/test/unit/language/test_pipeliner.py"),
            test_filter="test_pipeline_epilogue[1-0]",
        ),

        # --------------------------------------------------------------
        # 语言层 - 原子操作: atomic CAS / tensor 上 atomic RMW
        # 覆盖 AtomicCASOp / AtomicRMWOp lowering
        # --------------------------------------------------------------
        TestConfig(
            file_path=os.path.join(test_dir, "python/test/unit/language/test_core.py"),
            test_filter="test_atomic_cas[int32-1-None]",
        ),
        TestConfig(
            # 8x8 tensor 上的 atomic_min, 覆盖 AtomicRMWOp 张量路径
            file_path=os.path.join(test_dir, "python/test/unit/language/test_core.py"),
            test_filter="test_tensor_atomic_rmw_block[1]",
        ),

        # --------------------------------------------------------------
        # 语言层 - 控制流: scf.if / scf.while / scf.for(反向迭代)
        # --------------------------------------------------------------
        TestConfig(
            file_path=os.path.join(test_dir, "python/test/unit/language/test_core.py"),
            test_filter="test_if_else",
        ),
        TestConfig(
            file_path=os.path.join(test_dir, "python/test/unit/language/test_core.py"),
            test_filter="test_while",
        ),
        TestConfig(
            # 反向 for-loop (iv<0), 验证有符号归纳变量 lowering
            file_path=os.path.join(test_dir, "python/test/unit/language/test_core.py"),
            test_filter="test_for_iv[22-18--1]",
        ),

        # --------------------------------------------------------------
        # 语言层 - load 语义: masked load + padding (other=0)
        # --------------------------------------------------------------
        TestConfig(
            # size_diff=2 触发 mask, other=0 走 padding 路径
            file_path=os.path.join(test_dir, "python/test/unit/language/test_core.py"),
            test_filter="test_masked_load[1-float32-128-2-0]",
        ),

        # --------------------------------------------------------------
        # 语言层 - shape op: expand_dims (覆盖 ExpandDimsOp 多种形态)
        # --------------------------------------------------------------
        TestConfig(
            file_path=os.path.join(test_dir, "python/test/unit/language/test_core.py"),
            test_filter="test_expand_dims",
        ),

        # --------------------------------------------------------------
        # 语言层 - noinline 函数调用: 验证调用约定 lowering
        # --------------------------------------------------------------
        TestConfig(
            # call_graph 模式: 完整调用链 (kernel -> noinline_fn -> 普通 fn)
            file_path=os.path.join(test_dir, "python/test/unit/language/test_core.py"),
            test_filter="test_noinline[call_graph]",
        ),

        # --------------------------------------------------------------
        # Runtime 层 : kernel reuse (cache 复用), autotuner 基本流程,
        # launch metadata hook
        # --------------------------------------------------------------
        TestConfig(
            # 同一 kernel 多次 launch 只编译一次 (jit_cache_hook 计数)
            # 覆盖 cache hit 路径; 与 test_nochange 功能重合, 保留更强 hook 版本
            file_path=os.path.join(test_dir, "python/test/unit/runtime/test_cache.py"),
            test_filter="test_reuse",
        ),
        TestConfig(
            # @triton.autotune + 多 Config 选择, 验证 cache 按 key 区分
            file_path=os.path.join(test_dir, "python/test/unit/runtime/test_autotuner.py"),
            test_filter="test_kwargs[False]",
        ),
        TestConfig(
            # launch_enter_hook + launch_metadata, 验证 launcher 元数据回调
            file_path=os.path.join(test_dir, "python/test/unit/runtime/test_launch.py"),
            test_filter="test_metadata",
        ),

        # --------------------------------------------------------------
        # 基础设施 / 工具: linear layout 代数, filecheck 框架自检
        # --------------------------------------------------------------
        TestConfig(
            # LinearLayout.compose 基础代数, 验证 layout 推导工具
            file_path=os.path.join(test_dir, "python/test/unit/tools/test_linear_layout.py"),
            test_filter="test_compose",
        ),
        TestConfig(
            # filecheck 正向 smoke test, 保证 IR 自检框架可用
            file_path=os.path.join(test_dir, "python/test/unit/test_filecheck.py"),
            test_filter="test_filecheck_positive",
        ),

        # --------------------------------------------------------------
        # PPU AIU - load 类: 覆盖普通 2D load、越界 padding 和 block-pointer 形式
        # --------------------------------------------------------------
        TestConfig(
            # 普通 aiu_load (offsets/shape/strides 直接传入)
            file_path=os.path.join(test_dir, "python/test/unit/ppu/aiu/test_aiu_load.py"),
            test_filter="test_aiu_load[64-64-1024-1024-2-2]",
        ),
        TestConfig(
            # block-pointer (tl.make_block_ptr) 形式的 aiu_load
            file_path=os.path.join(test_dir, "python/test/unit/ppu/aiu/test_aiu_tensor_ptr.py"),
            test_filter="test_aiu_load[64-64-1024-1024-2-2]",
        ),

        # --------------------------------------------------------------
        # PPU AIU - dot
        # --------------------------------------------------------------
        TestConfig(
            # AIU fp16 matmul 主路径(AcceleratePPUMatmul)
            file_path=os.path.join(test_dir, "python/test/unit/ppu/aiu/test_aiu_dot.py"),
            test_filter="test_aiu_matmul[False-64-64-64-1024-1024-1024-2-2]",
        ),

        # --------------------------------------------------------------
        # PPU 端到端 - fused attention: flash-attention 完整 kernel,
        # 覆盖 causal mask + 通用 tl.dot 路径与 AIU 加速路径
        # --------------------------------------------------------------
        TestConfig(
            # 通用 (非 AIU) flash-attention: causal=True, Z=1 H=2 N_CTX=1024 HEAD_DIM=64
            file_path=os.path.join(test_dir, "python/test/unit/ppu/perf/06-fused-attention.py"),
            test_filter="test_op[True-1-2-1024-64]",
        ),
        TestConfig(
            # AIU fp16 flash-attention: 验证 aiu_load + aiu_dot 在完整 attention kernel 中的协作
            file_path=os.path.join(test_dir, "python/test/unit/ppu/perf/06-fused-attention-aiu.py"),
            test_filter="test_op_fp16[True-1-2-1024-64]",
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
            "  python triton_ci.py\n"
            "  python triton_ci.py -o result.xml --test-dir /path/to/repo\n"
            "  python triton_ci.py --keep-temp -v\n"
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

    args = parser.parse_args(argv)

    # ---- 加载测试配置 ----
    # 使用默认配置；如需自定义，可修改此处或通过外部配置文件加载
    test_configs: List[TestConfig] = get_default_test_configs(args.test_dir)

    if not test_configs:
        print("⚠️ 没有配置任何测试用例")

    print(f"📋 共 {len(test_configs)} 个测试用例待运行")
    print(f"📂 测试基础目录: {os.path.abspath(args.test_dir)}")
    print(f"📄 输出文件: {args.output}")

    # ---- 逐个运行测试 ----
    results: List[TestResult] = []
    for idx, cfg in enumerate(test_configs):
        result = run_single_test(cfg, idx, verbose=args.verbose)
        results.append(result)

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

    # ---- 返回退出码 ----
    has_failures = any(not r.passed for r in results)
    return 1 if has_failures else 0


if __name__ == "__main__":
    sys.exit(main())
