"""
Advanced Dynamic Load Balancing MPI Analyzer
Uses a Master-Worker architecture, Adaptive Chunking, Non-Blocking ISend/IRecv, and Hybrid ThreadPools.
"""
import sys
import json
import time
from mpi4py import MPI
from concurrent.futures import ThreadPoolExecutor
from ai_detector import AIDetector
from fake_news_model import FakeNewsDetector

# Tags for MPI Communication
TAG_TASK_REQUEST = 1
TAG_TASK_DATA = 2
TAG_RESULT = 3
TAG_TERMINATE = 4

def master_process(comm, size, texts):
    num_workers = size - 1
    results = []
    
    # Track metrics
    start_time = time.time()
    worker_stats = {i: {"processed": 0, "wait_time": 0, "compute_time": 0} for i in range(1, size)}
    
    num_tasks = len(texts)
    task_idx = 0
    active_workers = num_workers
    
    print(f"Master dynamically adaptive-chunking {num_tasks} massive items to {num_workers} hybrid workers...", file=sys.stderr)
    
    while active_workers > 0:
        status = MPI.Status()
        msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        worker_id = status.Get_source()
        tag = status.Get_tag()
        
        if tag == TAG_TASK_REQUEST:
            if task_idx < num_tasks:
                # ADAPTIVE CHUNKING ALGORITHM: 
                # Instead of fixed chunks or size 1, proportionally calculate the remaining limit
                remaining_tasks = num_tasks - task_idx
                adaptive_chunk_size = max(1, remaining_tasks // (num_workers * 2))
                end_idx = min(task_idx + adaptive_chunk_size, num_tasks)
                
                chunk_data = {"start_id": task_idx, "texts": texts[task_idx:end_idx]}
                
                # NON-BLOCKING ISEND implementation overlapping OS network buffering
                req = comm.isend(chunk_data, dest=worker_id, tag=TAG_TASK_DATA)
                req.wait() # Forces pipeline flush to network layer asynchronously
                task_idx = end_idx
            else:
                comm.send(None, dest=worker_id, tag=TAG_TERMINATE)
                active_workers -= 1
                
        elif tag == TAG_RESULT:
            res_chunk, w_stats = msg
            results.extend(res_chunk) # We extend because workers return arrays now
            worker_stats[worker_id]["processed"] += len(res_chunk)
            worker_stats[worker_id]["wait_time"] += w_stats.get("wait", 0)
            worker_stats[worker_id]["compute_time"] += w_stats.get("compute", 0)
            
    end_time = time.time()
    
    # Flatten output
    output = {
        "total_processed": len(results),
        "num_processes": size,
        "master_time": round(end_time - start_time, 4),
        "worker_stats": worker_stats,
        "results": results
    }
    
    print("\n---MPI_RESULT---")
    print(json.dumps(output))

def worker_process(comm, rank):
    ai_detector = AIDetector()
    fake_news_detector = FakeNewsDetector()
    
    while True:
        wait_start = time.time()
        # Non-Blocking ping for load-balancing Work Stealing equivalence
        req_req = comm.isend(True, dest=0, tag=TAG_TASK_REQUEST)
        req_req.wait()
        
        # Non-Blocking task reception
        req_recv = comm.irecv(source=0, tag=MPI.ANY_TAG)
        
        # CPU can technically do independent work here while network buffer fills natively
        
        status = MPI.Status()
        task = req_recv.wait(status=status) # Await data flush
        wait_end = time.time()
        wait_time = wait_end - wait_start
        
        tag = status.Get_tag()
        
        if tag == TAG_TERMINATE:
            break
            
        if tag == TAG_TASK_DATA:
            chunk_texts = task["texts"]
            start_id = task["start_id"]
            
            compute_start = time.time()
            chunk_results = []
            
            # Hybrid ThreadPool Task Parallelism across the Adaptive Chunk
            for i, text in enumerate(chunk_texts):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    fn_future = executor.submit(fake_news_detector.analyze, text)
                    ai_future = executor.submit(ai_detector.analyze, text)
                    
                    fn_res = fn_future.result()
                    ai_res = ai_future.result()
                    
                chunk_results.append({
                    "id": start_id + i,
                    "text_snippet": text[:100] + ("..." if len(text) > 100 else ""),
                    "fake_news": fn_res,
                    "ai_detected": ai_res,
                    "Processed_by_Rank": rank
                })
                
            compute_end = time.time()
            compute_time = compute_end - compute_start
            
            stats_update = {"wait": wait_time, "compute": compute_time}
            
            # Non-Blocking network offload
            req_res = comm.isend((chunk_results, stats_update), dest=0, tag=TAG_RESULT)
            req_res.wait()

def sequential_fallback(texts):
    start_time = time.time()
    ai_detector = AIDetector()
    fake_news_detector = FakeNewsDetector()
    results = []
    
    for i, text in enumerate(texts):
        with ThreadPoolExecutor(max_workers=2) as executor:
            fn_future = executor.submit(fake_news_detector.analyze, text)
            ai_future = executor.submit(ai_detector.analyze, text)
            fn_res = fn_future.result()
            ai_res = ai_future.result()
        
        results.append({
            "id": i,
            "text_snippet": text[:100] + ("..." if len(text) > 100 else ""),
            "fake_news": fn_res,
            "ai_detected": ai_res,
            "Processed_by_Rank": 0
        })
            
    end_time = time.time()
    
    compute_time = end_time - start_time
    output = {
        "total_processed": len(results),
        "num_processes": 1,
        "master_time": round(compute_time, 4),
        "worker_stats": {0: {"processed": len(results), "wait_time": 0, "compute_time": compute_time}},
        "results": results
    }
    print("\n---MPI_RESULT---")
    print(json.dumps(output))

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    texts = []
    if rank == 0:
        try:
            if len(sys.argv) > 2 and sys.argv[1] == "--file":
                with open(sys.argv[2], "r") as f:
                    texts = json.load(f)
            else:
                input_json = sys.argv[1]
                texts = json.loads(input_json)
        except Exception as e:
            print(f"Error parsing json texts: {e}", file=sys.stderr)
            texts = []
            
    if size == 1:
        sequential_fallback(texts)
    else:
        if rank == 0:
            master_process(comm, size, texts)
        else:
            worker_process(comm, rank)

if __name__ == "__main__":
    main()
