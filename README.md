# 🌐 TruthNet Lite

**TruthNet Lite** is an advanced, High-Performance Computing (HPC) research project mathematically designed to parallelize Data Science workloads across isolated processor cores. Conceived as a scalable, distributed memory text-classification pipeline, the system parses enormous payload sets through complex localized Machine Learning matrices analyzing textual geometries for AI presence and Disinformation confidence. 

It heavily utilizes `OpenMPI` arrays mapped directly onto `React` visualization streams to algorithmically present core scalability parameters (Efficiency, Amdahl's Law, Gustafson's limitations, and Karp-Flatt overhead calculations).

🚀 **Live Deployment Demo**: [https://truthnet-lite.onrender.com](https://truthnet-lite.onrender.com)

---

## ⚡ Active Computational Architecture
- **Distributed Memory Subnets (MPI)**: Core execution matrices are totally isolated per machine thread utilizing `mpi4py`. Evaluated via pure collective logic topologies (`comm.scatter()` and `comm.gather()`).
- **RAM Kaggle Caches (Data Parallelism)**: Leverages `KaggleHub` dynamically porting `240MB+` datasets dynamically into global cache allocations resolving strict IO memory bottlenecks natively.
- **RESTful Bridging**: Incorporates a massive `FastAPI` instance interacting concurrently with React utilizing virtual subprocesses binding memory natively from web domains onto the backend MPI compiler explicitly.

## 🛠 Prerequisites
- **Python 3.9+**
- **Node.js 22.x**
- **OpenMPI Library** (`libopenmpi-dev` or `brew install open-mpi`)

## 💻 Local Setup (macOS)
1. Navigate to the root directory and run the initialization script:
   ```bash
   bash setup_mac.sh
   ```
2. Activate the python virtual environment:
   ```bash
   source backend/truthnet-env/bin/activate
   ```
3. Boot the FastAPI engine (Port 8000):
   ```bash
   cd backend
   uvicorn app:app --reload
   ```
4. In an isolated logic terminal, execute the React array UI:
   ```bash
   cd frontend/truthnet-ui
   npm run dev
   ```

## 🐋 Cloud Deployment (Docker)
The system is entirely containerized bridging physical OpenMPI restrictions allowing explicit single monolithic hosting dynamically on SaaS bounds!
```bash
docker build -t truthnet-lite .
docker run -p 8000:8000 truthnet-lite
```

## 📊 Academic Features
- **Global Evaluation Metrics**: Calculates massive Aggregated Trust Scores mathematically evaluating hundreds of algorithmic fractions independently.
- **DOM Virtual Protection**: Restricts React processing limits natively strictly slicing mappings preventing CPU browser freezing bounds automatically.
- **Pure Thread-less Operations**: Overrides standard POSIX threads explicitly enforcing standard Synchronous single-core ML inferences actively completely eliminating `Python GIL (Global Interpreter Lock)` deadlocks. 
