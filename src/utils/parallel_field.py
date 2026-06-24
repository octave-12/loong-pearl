"""
真正的并行处理 - 为每个worker创建独立场实例
"""
import torch
import multiprocessing as mp
from multiprocessing import Pool, Queue
from typing import List, Callable, Any, Optional
import copy


class ParallelFieldProcessor:
    """并行场处理器，每个worker拥有独立场实例"""
    
    def __init__(
        self,
        field_factory: Callable[[], Any],
        num_workers: int = None,
        device: str = 'cpu'
    ):
        """
        Args:
            field_factory: 创建场实例的工厂函数
            num_workers: worker数量（默认CPU核心数）
            device: 设备类型
        """
        self.field_factory = field_factory
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)
        self.device = device
        self._workers = []
        self._task_queue = None
        self._result_queue = None
    
    def process_batch_parallel(
        self,
        inputs: List[str],
        process_func: Callable[[Any, str], str]
    ) -> List[str]:
        """并行处理一批输入
        
        Args:
            inputs: 输入列表
            process_func: 处理函数，接受(场实例, 输入)返回输出
        
        Returns:
            输出列表
        """
        if len(inputs) == 0:
            return []
        
        if len(inputs) == 1:
            field = self.field_factory()
            return [process_func(field, inputs[0])]
        
        chunk_size = max(1, len(inputs) // self.num_workers)
        chunks = [inputs[i:i + chunk_size] for i in range(0, len(inputs), chunk_size)]
        
        with Pool(self.num_workers) as pool:
            results = pool.map(
                _process_chunk_wrapper,
                [(self.field_factory, chunk, process_func) for chunk in chunks]
            )
        
        outputs = []
        for chunk_result in results:
            outputs.extend(chunk_result)
        
        return outputs


def _process_chunk_wrapper(args):
    """Pool.map的包装函数"""
    field_factory, chunk, process_func = args
    field = field_factory()
    return [process_func(field, item) for item in chunk]


class AsyncFieldProcessor:
    """异步场处理器，使用独立进程处理"""
    
    def __init__(
        self,
        field_factory: Callable[[], Any],
        num_workers: int = 2
    ):
        self.field_factory = field_factory
        self.num_workers = num_workers
        self._task_queue = Queue()
        self._result_queue = Queue()
        self._workers = []
        self._running = False
    
    def start(self):
        """启动worker进程"""
        if self._running:
            return
        
        for i in range(self.num_workers):
            worker = mp.Process(
                target=self._worker_loop,
                args=(i, self.field_factory, self._task_queue, self._result_queue),
                daemon=True
            )
            worker.start()
            self._workers.append(worker)
        
        self._running = True
    
    def stop(self):
        """停止worker进程"""
        if not self._running:
            return
        
        for _ in range(self.num_workers):
            self._task_queue.put(None)
        
        for worker in self._workers:
            worker.join(timeout=5)
        
        self._workers.clear()
        self._running = False
    
    @staticmethod
    def _worker_loop(worker_id, field_factory, task_queue, result_queue):
        """worker主循环"""
        field = field_factory()
        
        while True:
            task = task_queue.get()
            if task is None:
                break
            
            task_id, input_data, process_func = task
            try:
                result = process_func(field, input_data)
                result_queue.put((task_id, result, None))
            except Exception as e:
                result_queue.put((task_id, None, str(e)))
    
    def submit(self, task_id: int, input_data: str, process_func: Callable):
        """提交任务"""
        self._task_queue.put((task_id, input_data, process_func))
    
    def get_result(self, timeout: float = None) -> Optional[tuple]:
        """获取结果"""
        try:
            return self._result_queue.get(timeout=timeout)
        except:
            return None


class ThreadLocalFieldPool:
    """线程本地场池，每个线程拥有独立实例"""
    
    def __init__(self, field_factory: Callable[[], Any]):
        self.field_factory = field_factory
        self._local_instances = {}
        self._lock = mp.Lock()
    
    def get_instance(self, thread_id: int = None) -> Any:
        """获取当前线程的场实例"""
        import threading
        
        if thread_id is None:
            thread_id = threading.get_ident()
        
        with self._lock:
            if thread_id not in self._local_instances:
                self._local_instances[thread_id] = self.field_factory()
            return self._local_instances[thread_id]
    
    def clear(self):
        """清空所有实例"""
        with self._lock:
            self._local_instances.clear()
    
    def stats(self) -> dict:
        """统计信息"""
        with self._lock:
            return {
                "num_instances": len(self._local_instances),
                "thread_ids": list(self._local_instances.keys())
            }