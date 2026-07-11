"""CLI entry point for batch processing."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from watermark_app.core.pipeline import process


def main():
    parser = argparse.ArgumentParser(description='水印去除工具 CLI')
    parser.add_argument('input', help='输入图片路径')
    parser.add_argument('-o', '--output', default='', help='输出路径')
    parser.add_argument('-m', '--mode', default='corner',
                        choices=['template', 'corner', 'region', 'mask', 'auto'],
                        help='处理模式')
    parser.add_argument('-r', '--reference', default='', help='参考图片 (template模式)')
    parser.add_argument('-c', '--corner', default='bottom-right',
                        choices=['top-left', 'top-right', 'bottom-left', 'bottom-right'],
                        help='角落位置 (corner模式)')
    parser.add_argument('-b', '--backend', default='lama',
                        choices=['lama', 'opencv'],
                        help='修复后端')
    parser.add_argument('-s', '--scan-pct', type=float, default=0.18, help='扫描比例')
    parser.add_argument('-p', '--padding', type=int, default=15, help='掩膜扩张')
    args = parser.parse_args()

    result = process(
        input_path=args.input,
        output_path=args.output,
        mode=args.mode,
        reference_path=args.reference,
        corner=args.corner,
        backend=args.backend,
        scan_pct=args.scan_pct,
        padding=args.padding,
    )

    print(f"状态: {result.status}")
    if result.output_path:
        print(f"输出: {result.output_path}")
    if result.warnings:
        for w in result.warnings:
            print(f"警告: {w}")
    print(f"耗时: {result.elapsed:.2f}s")


if __name__ == '__main__':
    main()
