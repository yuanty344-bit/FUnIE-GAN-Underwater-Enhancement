"""
批量处理器模块

提供批量图像增强处理功能，
支持多线程并行处理和进度跟踪。
"""

import os
import logging
import time
from pathlib import Path
from typing import Optional, List, Callable, Tuple, Dict, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
import json

import numpy as np

from ..utils.file_io import (
    read_image, write_image, get_image_files,
    ensure_directories_exist
)
from ..core.enhancer import ImageEnhancer, EnhancementMode
from ..core.metrics import ImageMetrics

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """处理模式"""
    SEQUENTIAL = 'sequential'  # 顺序处理
    PARALLEL = 'parallel'  # 并行处理
    ADAPTIVE = 'adaptive'  # 自适应（根据文件大小）


@dataclass
class ProcessingResult:
    """单张图像处理结果"""
    input_path: Path
    output_path: Optional[Path]
    success: bool
    error: Optional[str]
    elapsed_time: float
    metrics: Optional[dict] = None


@dataclass
class BatchStatistics:
    """批量处理统计"""
    total_files: int
    success_count: int
    error_count: int
    skipped_count: int
    total_time: float
    average_time: float
    success_rate: float
    results: List[ProcessingResult]


class BatchProcessor:
    """
    批量图像处理器
    
    支持批量处理目录中的图像文件，
    提供进度回调和详细的结果报告。
    
    Attributes:
        input_dir: 输入目录
        output_dir: 输出目录
        enhancer: 图像增强器
        mode: 处理模式
        skip_existing: 跳过已处理文件
        
    Example:
        >>> processor = BatchProcessor('./input', './output', enhancer)
        >>> processor.process(progress_callback=my_callback)
        >>> stats = processor.get_statistics()
    """
    
    def __init__(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        enhancer: ImageEnhancer,
        mode: ProcessingMode = ProcessingMode.PARALLEL,
        max_workers: int = 4,
        skip_existing: bool = True,
        compute_metrics: bool = False,
        preserve_structure: bool = True
    ):
        """
        初始化批量处理器
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            enhancer: 图像增强器实例
            mode: 处理模式
            max_workers: 最大工作线程数
            skip_existing: 跳过已存在的输出文件
            compute_metrics: 是否计算质量指标
            preserve_structure: 是否保持目录结构
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.enhancer = enhancer
        self.mode = mode
        self.max_workers = max_workers
        self.skip_existing = skip_existing
        self.compute_metrics = compute_metrics
        self.preserve_structure = preserve_structure
        
        # 运行时状态
        self._is_running = False
        self._is_cancelled = False
        self._results: List[ProcessingResult] = []
        self._skipped_count = 0
        self._metrics = ImageMetrics()
        
        # 确保输出目录存在
        ensure_directories_exist([self.output_dir])
        
        logger.info(f"BatchProcessor初始化: input={input_dir}, output={output_dir}")
    
    def process(
        self,
        progress_callback: Optional[Callable] = None,
        filter_extensions: Optional[List[str]] = None
    ) -> BatchStatistics:
        """
        执行批量处理
        
        Args:
            progress_callback: 进度回调函数，签名为 callback(current, total, message)
            filter_extensions: 文件扩展名过滤器
            
        Returns:
            BatchStatistics 统计结果
        """
        # 获取文件列表
        image_files = get_image_files(self.input_dir, recursive=True)
        
        if not image_files:
            logger.warning(f"在 {self.input_dir} 中未找到图像文件")
            return self._create_statistics(0)
        
        logger.info(f"找到 {len(image_files)} 个图像文件")
        
        # 初始化状态
        self._is_running = True
        self._is_cancelled = False
        self._results = []
        self._skipped_count = 0
        
        start_time = time.time()
        
        # 根据模式选择处理方式
        if self.mode == ProcessingMode.SEQUENTIAL:
            self._process_sequential(image_files, progress_callback)
        elif self.mode == ProcessingMode.PARALLEL:
            self._process_parallel(image_files, progress_callback)
        elif self.mode == ProcessingMode.ADAPTIVE:
            # 自适应模式：大文件顺序处理，小文件并行
            self._process_adaptive(image_files, progress_callback)
        
        total_time = time.time() - start_time
        
        # 生成统计
        stats = self._create_statistics(
            len(image_files),
            total_time
        )
        
        self._is_running = False
        
        logger.info(f"批量处理完成: 成功 {stats.success_count}/{stats.total_files}, "
                   f"耗时 {total_time:.2f}秒")
        
        return stats
    
    def _process_sequential(
        self,
        image_files: List[Path],
        progress_callback: Optional[Callable]
    ) -> None:
        """
        顺序处理
        
        Args:
            image_files: 图像文件列表
            progress_callback: 进度回调
        """
        total = len(image_files)
        
        for i, file_path in enumerate(image_files):
            if self._is_cancelled:
                break
            
            # 报告进度
            message = f"处理中: {file_path.name}"
            if progress_callback:
                progress_callback(i + 1, total, message)
            
            # 处理单个文件
            result = self._process_single_file(file_path)
            self._results.append(result)
    
    def _process_parallel(
        self,
        image_files: List[Path],
        progress_callback: Optional[Callable]
    ) -> None:
        """
        并行处理
        
        Args:
            image_files: 图像文件列表
            progress_callback: 进度回调
        """
        total = len(image_files)
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(self._process_single_file, file_path): file_path
                for file_path in image_files
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_file):
                if self._is_cancelled:
                    # 取消所有待处理任务
                    for f in future_to_file:
                        f.cancel()
                    break
                
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    self._results.append(result)
                except Exception as e:
                    logger.error(f"处理异常 {file_path}: {e}")
                    self._results.append(ProcessingResult(
                        input_path=file_path,
                        output_path=None,
                        success=False,
                        error=str(e),
                        elapsed_time=0.0
                    ))
                
                completed += 1
                
                if progress_callback:
                    message = f"处理中: {file_path.name} ({completed}/{total})"
                    progress_callback(completed, total, message)
    
    def _process_adaptive(
        self,
        image_files: List[Path],
        progress_callback: Optional[Callable]
    ) -> None:
        """
        自适应处理
        
        根据文件大小决定处理方式：
        - 大文件(>5MB)：顺序处理，避免内存问题
        - 小文件(≤5MB)：并行处理，提高速度
        
        Args:
            image_files: 图像文件列表
            progress_callback: 进度回调
        """
        # 按大小分类
        large_files = []
        small_files = []
        
        for file_path in image_files:
            size = file_path.stat().st_size
            if size > 5 * 1024 * 1024:  # > 5MB
                large_files.append(file_path)
            else:
                small_files.append(file_path)
        
        total = len(image_files)
        completed = 0
        
        # 先处理大文件（顺序）
        for file_path in large_files:
            if self._is_cancelled:
                break
            
            if progress_callback:
                message = f"处理大文件: {file_path.name}"
                progress_callback(completed + 1, total, message)
            
            result = self._process_single_file(file_path)
            self._results.append(result)
            completed += 1
        
        # 再处理小文件（并行）
        if not self._is_cancelled and small_files:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(self._process_single_file, fp): fp
                    for fp in small_files
                }
                
                for future in as_completed(future_to_file):
                    if self._is_cancelled:
                        for f in future_to_file:
                            f.cancel()
                        break
                    
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        self._results.append(result)
                    except Exception as e:
                        self._results.append(ProcessingResult(
                            input_path=file_path,
                            output_path=None,
                            success=False,
                            error=str(e),
                            elapsed_time=0.0
                        ))
                    
                    completed += 1
                    
                    if progress_callback:
                        message = f"处理中: {file_path.name} ({completed}/{total})"
                        progress_callback(completed, total, message)
    
    def _process_single_file(self, file_path: Path) -> ProcessingResult:
        """
        处理单个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            ProcessingResult 处理结果
        """
        start_time = time.time()
        
        # 确定输出路径
        output_path = self._get_output_path(file_path)
        
        # 检查是否跳过
        if self.skip_existing and output_path.exists():
            logger.debug(f"跳过已存在: {output_path}")
            self._skipped_count += 1
            return ProcessingResult(
                input_path=file_path,
                output_path=output_path,
                success=True,
                error=None,
                elapsed_time=time.time() - start_time
            )
        
        try:
            # 读取图像
            image = read_image(file_path)
            
            # 增强处理
            enhanced = self.enhancer.enhance(image)
            
            # 保存结果
            write_image(enhanced, output_path)
            
            # 计算指标
            metrics = None
            if self.compute_metrics:
                metrics = self._metrics.evaluate(image, enhanced)
            
            elapsed = time.time() - start_time
            
            logger.debug(f"处理完成: {file_path.name} -> {output_path.name} ({elapsed:.2f}s)")
            
            return ProcessingResult(
                input_path=file_path,
                output_path=output_path,
                success=True,
                error=None,
                elapsed_time=elapsed,
                metrics=metrics.to_dict() if metrics else None
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"处理失败: {file_path.name}, 错误: {e}")
            
            return ProcessingResult(
                input_path=file_path,
                output_path=None,
                success=False,
                error=str(e),
                elapsed_time=elapsed
            )
    
    def _get_output_path(self, input_path: Path) -> Path:
        """
        获取输出文件路径
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            输出文件路径
        """
        if self.preserve_structure:
            # 保持相对目录结构
            try:
                relative = input_path.relative_to(self.input_dir)
                return self.output_dir / f"{relative.stem}_enhanced{relative.suffix}"
            except ValueError:
                # 不在同一目录树下
                return self.output_dir / f"{input_path.stem}_enhanced{input_path.suffix}"
        else:
            return self.output_dir / f"{input_path.stem}_enhanced{input_path.suffix}"
    
    def _create_statistics(
        self,
        total_files: int,
        total_time: float
    ) -> BatchStatistics:
        """
        创建统计结果
        
        Args:
            total_files: 总文件数
            total_time: 总耗时
            
        Returns:
            BatchStatistics 统计对象
        """
        success = sum(1 for r in self._results if r.success)
        errors = sum(1 for r in self._results if not r.success)
        skipped = self._skipped_count if hasattr(self, '_skipped_count') else 0
        
        return BatchStatistics(
            total_files=total_files,
            success_count=success,
            error_count=errors,
            skipped_count=skipped,
            total_time=total_time,
            average_time=total_time / total_files if total_files > 0 else 0,
            success_rate=success / total_files if total_files > 0 else 0,
            results=self._results
        )
    
    def cancel(self) -> None:
        """取消处理"""
        self._is_cancelled = True
        logger.info("批量处理已取消")
    
    def stop(self) -> bool:
        """
        停止处理（用于回调中）
        
        Returns:
            是否应该停止
        """
        return self._is_cancelled
    
    @property
    def is_running(self) -> bool:
        """检查是否正在运行"""
        return self._is_running
    
    def get_statistics(self) -> Optional[BatchStatistics]:
        """
        获取处理统计
        
        Returns:
            统计结果，处理完成前返回None
        """
        if not self._results:
            return None
        
        total_time = sum(r.elapsed_time for r in self._results)
        return self._create_statistics(len(self._results), total_time)
    
    def export_results(self, output_path: Union[str, Path]) -> bool:
        """
        导出处理结果为JSON
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            导出是否成功
        """
        stats = self.get_statistics()
        
        if stats is None:
            return False
        
        # 构建导出数据
        export_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'input_directory': str(self.input_dir),
            'output_directory': str(self.output_dir),
            'statistics': {
                'total_files': stats.total_files,
                'success_count': stats.success_count,
                'error_count': stats.error_count,
                'skipped_count': stats.skipped_count,
                'total_time': stats.total_time,
                'average_time': stats.average_time,
                'success_rate': stats.success_rate
            },
            'results': [
                {
                    'input': str(r.input_path),
                    'output': str(r.output_path) if r.output_path else None,
                    'success': r.success,
                    'error': r.error,
                    'elapsed_time': r.elapsed_time,
                    'metrics': r.metrics
                }
                for r in stats.results
            ]
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"结果已导出: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"导出失败: {e}")
            return False
    
    def export_failed_files(self, output_path: Union[str, Path]) -> bool:
        """
        导出失败文件列表
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            导出是否成功
        """
        failed = [r for r in self._results if not r.success]
        
        if not failed:
            return True
        
        export_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'failed_count': len(failed),
            'failed_files': [
                {
                    'path': str(r.input_path),
                    'error': r.error
                }
                for r in failed
            ]
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"失败文件列表已导出: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"导出失败: {e}")
            return False
