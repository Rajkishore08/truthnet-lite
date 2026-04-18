"""
Pure Synchronous Scatter/Gather MPI Analyzer
Adheres strictly to core Master -> Map Array -> Gather operations without complex network loops.
"""
import sys
import json
import time
import math
from mpi4py import MPI
from concurrent.futures import ThreadPoolExecutor
from ai_detector import AIDetector
from fake_news_model import FakeNewsDetector

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    start_time = time.time()
    
    # 1. Master Sequence: Parse Array & Split
    chunks = None
    if rank == 0:
        texts = []
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
            
        n_items = len(texts)
        if n_items == 0:
            chunks = [[] for _ in range(size)]
        else:
            # Slicing datasets perfectly evenly
            chunk_size = math.ceil(n_items / size)
            chunks = [texts[i:i + chunk_size] for i in range(0, n_items, chunk_size)]
            
        # Zero-pad arrays if nodes heavily exceed payload fractions naturally avoiding native scatter bounds
        while len(chunks) < size:
            chunks.append([])
            
    # 2. SCATTER: Massive broadcast segmentating matrix synchronously across subnet instances!
    local_chunk = comm.scatter(chunks, root=0)
    
    # 3. COMPUTE: Isolated processing
    ai_detector = AIDetector()
    fake_news_detector = FakeNewsDetector()
    
    local_results = []
    compute_start = time.time()
    
    for text in local_chunk:
        with ThreadPoolExecutor(max_workers=2) as executor:
            fn_future = executor.submit(fake_news_detector.analyze, text)
            ai_future = executor.submit(ai_detector.analyze, text)
            
            fn_res = fn_future.result()
            ai_res = ai_future.result()
            
        local_results.append({
            "text_snippet": text[:100] + ("..." if len(text) > 100 else ""),
            "fake_news": fn_res,
            "ai_detected": ai_res,
            "Processed_by_Rank": rank
        })
        
    compute_end = time.time()
    compute_time = compute_end - compute_start
    
    # 4. GATHER: Subnets natively aggregate matrices into massive synchronous return array!
    gathered_results = comm.gather(local_results, root=0)
    gathered_metrics = comm.gather({"compute_time": compute_time, "processed": len(local_chunk)}, root=0)
    
    # 5. Master Sequence: Flatten output into core WebHook framework payload
    if rank == 0:
        flat_results = []
        worker_stats = {}
        for r, res_array in enumerate(gathered_results):
            for res in res_array:
                res["id"] = len(flat_results)
                flat_results.append(res)
                
            worker_stats[r] = {
                "processed": gathered_metrics[r]["processed"],
                "wait_time": 0, # Sync scatter bypasses queue waiting algorithms organically
                "compute_time": gathered_metrics[r]["compute_time"]
            }
            
        output = {
            "total_processed": len(flat_results),
            "num_processes": size,
            "master_time": round(time.time() - start_time, 4),
            "worker_stats": worker_stats,
            "results": flat_results
        }
        
        print("\n---MPI_RESULT---")
        print(json.dumps(output))

if __name__ == "__main__":
    main()
