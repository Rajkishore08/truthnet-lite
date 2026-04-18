import { useState } from 'react';
import axios from 'axios';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
} from 'chart.js';
import { Bar, Doughnut } from 'react-chartjs-2';
import FloatingLines from './FloatingLines';
import CountUp from './CountUp';

// Register ChartJS plugins
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const DEV_API_URL = 'http://localhost:8000/analyze';
const KAGGLE_API_URL = 'http://localhost:8000/analyze_kaggle';

function App() {
  const [text, setText] = useState('');
  const [numProcesses, setNumProcesses] = useState(4);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [performanceHistory, setPerformanceHistory] = useState([]);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    if (!text.trim()) return;
    
    setLoading(true);
    setError(null);
    
    const API_BASE = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') ? 'http://127.0.0.1:8000' : '';
    
    try {
      // Split text into lines to simulate multiple articles for batch processing if needed
      const articles = text.split('\n').filter(t => t.trim().length > 0);
      
      const response = await axios.post(`${API_BASE}/analyze`, {
        texts: articles,
        num_processes: parseInt(numProcesses)
      });
      
      const data = response.data;
      setResults(data.mpi_output);
      
      // Update performance history
      setPerformanceHistory(prev => [
        ...prev, 
        { processes: numProcesses, time: data.execution_time_seconds, articles: articles.length }
      ]);
      
    } catch (error) {
      console.error("Error analyzing:", error);
      setError("Analysis failed. Make sure the FastAPI backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyzeKaggle = async () => {
    setLoading(true);
    setError(null);
    
    const API_BASE = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') ? 'http://127.0.0.1:8000' : '';

    try {
      const response = await axios.get(`${API_BASE}/analyze_kaggle?limit=1000&np=${numProcesses}`);
      const data = response.data;
      setResults(data.mpi_output);
      
      setPerformanceHistory(prev => [
        ...prev, 
        { processes: numProcesses, time: data.execution_time_seconds, articles: 1000 }
      ]);
    } catch (error) {
      console.error("Error analyzing:", error);
      alert("Kaggle Benchmark failed.");
    } finally {
      setLoading(false);
    }
  };

  const performanceData = {
    labels: performanceHistory.map((h, i) => `Run ${i+1} (${h.processes} cores)`),
    datasets: [
      {
        label: 'Execution Time (s)',
        data: performanceHistory.map(h => h.time),
        backgroundColor: 'rgba(6, 182, 212, 0.6)',
        borderColor: 'rgba(6, 182, 212, 1)',
        borderWidth: 1,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: '#f8fafc' } },
      title: { display: true, text: 'Parallel Scaling Performance', color: '#f8fafc' },
    },
    scales: {
      y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.1)' } },
      x: { ticks: { color: '#94a3b8' }, grid: { display: false } }
    }
  };

  // Node Telemetry Data Prep
  let telemetryData = null;
  if (results && results?.worker_stats) {
    const ranks = Object.keys(results.worker_stats);
    const usages = ranks.map(r => results.worker_stats[r].processed);
    
    telemetryData = {
      labels: ranks.map(r => `Rank ${r}`),
      datasets: [
        {
          data: usages,
          backgroundColor: [
            'rgba(0, 133, 239, 0.7)',
            'rgba(0, 255, 107, 0.7)',
            'rgba(139, 92, 246, 0.7)',
            'rgba(239, 68, 68, 0.7)',
            'rgba(245, 158, 11, 0.7)',
            'rgba(16, 185, 129, 0.7)'
          ],
          borderColor: 'rgba(0,0,0,0.2)',
          borderWidth: 1,
        }
      ]
    };
  }

  const doughnutOpts = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'right', labels: { color: '#f8fafc' } },
      title: { display: true, text: 'Worker Node Dispatch Matrix', color: '#f8fafc' },
    }
  };

  // Advanced Insights Prep
  let insights = null;
  if (results && results?.worker_stats) {
    const ranks = Object.keys(results.worker_stats);
    let masterTime = results.master_time || 0;
    
    let totalCompute = 0;
    let totalWait = 0;
    
    ranks.forEach(r => {
      totalCompute += results.worker_stats[r].compute_time;
      totalWait += results.worker_stats[r].wait_time;
    });

    const T1 = totalCompute; // Theoretical Serial Time is the sum of isolated CPU computes
    const Tp = masterTime;   // Actual Parallel Time including communication overhead
    
    let speedup = 0;
    let efficiency = 0;
    let karpFlatt = 0;
    let gustafson = 0;
    const P = 0.98; // Estimated strictly parallelizable fraction
    
    if (Tp > 0) {
      speedup = (T1 / Tp).toFixed(2);
      efficiency = ((speedup / results.num_processes) * 100).toFixed(1);
      
      const N = results.num_processes;
      if (N > 1) {
        karpFlatt = (((1/speedup) - (1/N)) / (1 - (1/N))).toFixed(4);
      } else {
        karpFlatt = 0;
      }
      
      gustafson = (N - (1 - P) * (N - 1)).toFixed(2);
    }

    insights = {
      speedup,
      efficiency,
      karpFlatt,
      gustafson,
      gustafson,
      masterTime: masterTime.toFixed(2),
      activeWorkers: ranks.length
    };
    
    let totalTrustSum = 0;
    let fakeCount = 0;
    let aiCount = 0;
    
    if(results.results && results.results.length > 0) {
      results.results.forEach(res => {
         let fake_score = res.fake_news.prediction === 'Real News' ? (res.fake_news.confidence || 0) : (100 - (res.fake_news.confidence || 100));
         let ai_score = 100 - (res.ai_detected.score || 0);
         if(isNaN(fake_score)) fake_score = 50;
         if(isNaN(ai_score)) ai_score = 50;
         totalTrustSum += (fake_score + ai_score) / 2;
         
         if (res.fake_news.prediction === 'Fake News') fakeCount++;
         if (res.ai_detected.prediction === 'AI Generated') aiCount++;
      });
      
      insights.globalTrust = (totalTrustSum / results.results.length).toFixed(1);
      insights.fakeCount = fakeCount;
      insights.aiCount = aiCount;
      
      let glColor = 'var(--accent-fake)';
      if (insights.globalTrust > 80) glColor = 'var(--accent-real)';
      else if (insights.globalTrust > 50) glColor = '#f59e0b';
      insights.globalColor = glColor;
    }
  }

  return (
    <>
      <div style={{ position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh', zIndex: -1 }}>
        <FloatingLines 
          enabledWaves={['top', 'middle', 'bottom']}
          lineCount={[10, 15, 20]}
          lineDistance={[8, 6, 4]}
          bendRadius={5.0}
          bendStrength={-0.5}
          interactive={true}
          parallax={true}
          linesGradient={['#0085ef', '#000000', '#00ff6b']}
        />
      </div>
      <div className="app-container">
        <header>
        <h1>TruthNet Lite</h1>
        <p className="subtitle" style={{fontSize: '1.2rem', color: '#cbd5e1'}}>A Hybrid MPI-Based Parallel System for Large-Scale Fake News Detection</p>
      </header>


      <div className="glass-panel input-section">
        <h2>Input Articles</h2>
        <p className="text-muted" style={{fontSize: '0.9rem', marginBottom: '-0.5rem'}}>
          Enter text to analyze. Separate multiple independent articles by a new line to leverage MPI chunking.
        </p>
        <textarea 
          placeholder="Paste news articles or suspicious text here..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        ></textarea>
        
        <div className="controls">
          <div className="process-select" style={{ display: 'flex', alignItems: 'center', marginBottom: '1.5rem'}}>
            <label style={{ marginRight: '1rem', color: '#cbd5e1', fontWeight: 'bold' }}>MPI Workers:</label>
            <div style={{ display: 'inline-flex', background: 'rgba(255,255,255,0.05)', borderRadius: '8px', overflow: 'hidden', border: '1px solid rgba(255,255,255,0.1)', flexShrink: 0 }}>
              {["1", "2", "4", "8"].map(val => (
                <button
                  key={val}
                  onClick={() => setNumProcesses(val)}
                  style={{
                    padding: '8px 16px',
                    background: String(numProcesses) === val ? 'var(--accent-blue)' : 'transparent',
                    color: String(numProcesses) === val ? '#fff' : '#94a3b8',
                    border: 'none',
                    borderRight: val !== "8" ? '1px solid rgba(255,255,255,0.1)' : 'none',
                    cursor: 'pointer',
                    fontWeight: String(numProcesses) === val ? 'bold' : 'normal',
                    transition: 'all 0.2s ease',
                    minWidth: '90px'
                  }}
                >
                  {val === "1" ? "1 Core" : `${val} Cores`}
                </button>
              ))}
            </div>
          </div>
          <button 
            className="btn-primary" 
            onClick={handleAnalyze} 
            disabled={loading || !text.trim()}
          >
            {loading ? <><span className="spinner"></span>Processing...</> : 'Analyze with MPI'}
          </button>
          <button 
            className="btn-primary" 
            onClick={handleAnalyzeKaggle} 
            disabled={loading}
            style={{background: 'linear-gradient(135deg, #f59e0b, #ef4444)'}}
          >
            {loading ? <><span className="spinner"></span>Benchmarking...</> : 'Kaggle Benchmark x1000'}
          </button>
        </div>
      </div>

      {results && (
        <div className="results-container">
          <div className="glass-panel">
            <h2>Analysis Results</h2>
            {insights && (
              <div style={{
                display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: '1rem', 
                marginBottom: '1rem', background: 'rgba(255,255,255,0.05)', 
                padding: '1rem', borderRadius: '12px'
              }}>
                <div>
                  <p style={{fontSize: '0.85rem', color: '#94a3b8'}}>Speedup ($S = T_s / T_p$)</p>
                  <p style={{fontSize: '1.4rem', fontWeight: 'bold', color: 'var(--accent-green)'}}>{insights.speedup}x</p>
                </div>
                <div>
                  <p style={{fontSize: '0.85rem', color: '#94a3b8'}}>Efficiency ($E = S / p$)</p>
                  <p style={{fontSize: '1.4rem', fontWeight: 'bold', color: 'var(--accent-blue)'}}>{insights.efficiency}%</p>
                </div>
                <div>
                  <p style={{fontSize: '0.85rem', color: '#94a3b8'}}>Karp-Flatt Metric ($e$)</p>
                  <p style={{fontSize: '1.4rem', fontWeight: 'bold', color: '#f59e0b'}}>{insights.karpFlatt}</p>
                </div>
                <div>
                  <p style={{fontSize: '0.85rem', color: '#94a3b8'}}>Gustafson's Law</p>
                  <p style={{fontSize: '1.4rem', fontWeight: 'bold', color: '#8b5cf6'}}>{insights.gustafson}x</p>
                </div>
              </div>
            )}
            
            {insights?.globalTrust && (
              <div style={{
                background: 'rgba(0,0,0,0.2)', padding: '1.5rem', borderRadius: '12px',
                borderLeft: `4px solid ${insights.globalColor}`, marginBottom: '1.5rem',
                display: 'flex', justifyContent: 'space-between', alignItems: 'center'
              }}>
                <div>
                  <h3 style={{color: insights.globalColor, fontSize: '2.5rem', margin: 0, lineHeight: '1'}}>
                    <CountUp
                      from={0}
                      to={parseFloat(insights.globalTrust)}
                      direction="up"
                      duration={1.5}
                    />
                  </h3>
                  <p style={{color: '#cbd5e1', margin: '5px 0 0 0', fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '2px'}}>Global Trust Score</p>
                </div>
                
                <div style={{display: 'flex', gap: '2rem', textAlign: 'center'}}>
                    <div>
                        <p style={{fontSize: '1.8rem', color: 'var(--accent-fake)', margin: 0, fontWeight: 'bold'}}>
                            <CountUp from={0} to={insights.fakeCount} separator="," duration={1.2} />
                        </p>
                        <p style={{fontSize: '0.8rem', color: '#94a3b8', margin: 0}}>Total Fake</p>
                    </div>
                    <div>
                        <p style={{fontSize: '1.8rem', color: '#ef4444', margin: 0, fontWeight: 'bold'}}>
                            <CountUp from={0} to={insights.aiCount} separator="," duration={1.2} />
                        </p>
                        <p style={{fontSize: '0.8rem', color: '#94a3b8', margin: 0}}>Total AI</p>
                    </div>
                    <div>
                        <p style={{fontSize: '1.8rem', color: 'var(--accent-blue)', margin: 0, fontWeight: 'bold'}}>
                            <CountUp from={0} to={parseFloat(insights.masterTime)} duration={1.5} />s
                        </p>
                        <p style={{fontSize: '0.8rem', color: '#94a3b8', margin: 0}}>Total Execution Time</p>
                    </div>
                </div>
              </div>
            )}
            
            <p className="text-muted" style={{fontSize: '0.9rem', marginBottom: '1rem'}}>
              Total processed articles: <CountUp from={0} to={results.total_processed} separator="," duration={1} />
            </p>
            
            <div style={{ maxHeight: '500px', overflowY: 'auto', paddingRight: '10px', marginTop: '1.5rem' }}>
              <p style={{fontSize: '0.8rem', color: '#94a3b8', textTransform: 'uppercase', marginBottom: '10px'}}>Displaying Log Preview (Top 100 entries)</p>
              {results.results.slice(0, 100).map((res, i) => {
                let fake_score = res.fake_news.prediction === 'Real News' ? (res.fake_news.confidence || 0) : (100 - (res.fake_news.confidence || 100));
                let ai_score = 100 - (res.ai_detected.score || 0); // score is 0-100 where 100 is highly AI generated
                if(isNaN(fake_score)) fake_score = 50;
                if(isNaN(ai_score)) ai_score = 50;
                const trustScore = ((fake_score + ai_score) / 2).toFixed(1);
                
                let trustColor = 'var(--accent-fake)';
                if (trustScore > 80) trustColor = 'var(--accent-real)';
                else if (trustScore > 50) trustColor = '#f59e0b';
                
                return (
                  <div key={i} className="result-card" style={{position: 'relative'}}>
                    <div style={{position: 'absolute', top: '15px', right: '15px', textAlign: 'right'}}>
                        <h4 style={{fontSize: '2rem', color: trustColor, lineHeight: '0.9', margin: 0}}>{trustScore}</h4>
                        <span style={{fontSize: '0.7rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '1px'}}>Trust Score</span>
                    </div>
                    
                    <p style={{marginBottom: '15px', fontSize: '0.95rem', color: '#cbd5e1', paddingRight: '60px', fontStyle: 'italic'}}>
                      "{res.text_snippet}"
                    </p>
                    <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap'}}>
                      <span className={`badge ${res.fake_news.prediction === 'Fake News' ? 'fake' : 'real'}`}>
                        {res.fake_news.prediction} {res.fake_news.confidence && `(${res.fake_news.confidence}%)`}
                      </span>
                      <span className={`badge ${res.ai_detected.prediction === 'AI Generated' ? 'ai' : 'human'}`}>
                        {res.ai_detected.prediction} {res.ai_detected.score && `(${(res.ai_detected.score).toFixed(1)}%)`}
                      </span>
                      <span style={{fontSize: '0.85rem', color: '#64748b', marginLeft: 'auto', background: 'rgba(255,255,255,0.05)', padding: '4px 8px', borderRadius: '4px'}}>
                        Threaded by MPI Rank {res.Processed_by_Rank}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
          
          <div className="glass-panel">
            <h2>Performance Benchmark</h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
              <div className="chart-container">
                {performanceHistory.length > 0 ? (
                  <Bar options={chartOptions} data={performanceData} />
                ) : (
                  <p>Run multiple tests to see scaling.</p>
                )}
              </div>

              {telemetryData && (
                <div className="chart-container" style={{borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '1rem' }}>
                  <Doughnut data={telemetryData} options={doughnutOpts} />
                </div>
              )}
            </div>
          </div>
        </div>
      )}
      
      <div className="glass-panel" style={{marginBottom: '2rem', background: 'linear-gradient(135deg, rgba(15,23,42,0.8), rgba(15,23,42,0.95))', borderColor: 'rgba(0,133,239,0.3)', boxShadow: '0 0 40px rgba(0,0,0,0.5)'}}>
        <h2 style={{borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '1rem'}}>⚡ Active Computational Architecture</h2>
        <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1.2rem', marginTop: '1.2rem'}}>
          <div className="result-card" style={{borderLeftColor: 'var(--accent-blue)', marginTop: 0, background: 'rgba(0,133,239,0.05)'}}>
            <h3 style={{color: 'var(--accent-blue)', marginBottom: '0.6rem', fontSize: '1.1rem'}}>🌐 Distributed Memory (MPI)</h3>
            <p style={{fontSize: '0.9rem', color: '#cbd5e1', lineHeight: '1.4'}}>Physical memory completely isolated per core. Worker Ranks operate independently scaling effortlessly across massive computing clusters.</p>
          </div>
          <div className="result-card" style={{borderLeftColor: 'var(--accent-green)', marginTop: 0, background: 'rgba(0,255,107,0.05)'}}>
            <h3 style={{color: 'var(--accent-green)', marginBottom: '0.6rem', fontSize: '1.1rem'}}>⚖️ Pure Scatter & Gather</h3>
            <p style={{fontSize: '0.9rem', color: '#cbd5e1', lineHeight: '1.4'}}>Rank 0 seamlessly segments massive array workloads recursively into perfect fractions synchronously broadcast utilizing comm.scatter().</p>
          </div>
          <div className="result-card" style={{borderLeftColor: '#f59e0b', marginTop: 0, background: 'rgba(245,158,11,0.05)'}}>
            <h3 style={{color: '#f59e0b', marginBottom: '0.6rem', fontSize: '1.1rem'}}>🔀 Data Parallelism</h3>
            <p style={{fontSize: '0.9rem', color: '#cbd5e1', lineHeight: '1.4'}}>Petabytes of Kaggle corpus fragments explicitly scattered directly from RAM into remote mathematical topologies!</p>
          </div>
          <div className="result-card" style={{borderLeftColor: '#8b5cf6', marginTop: 0, background: 'rgba(139,92,246,0.05)'}}>
            <h3 style={{color: '#8b5cf6', marginBottom: '0.6rem', fontSize: '1.1rem'}}>⚡ Hybrid Shared-Memory</h3>
            <p style={{fontSize: '0.9rem', color: '#cbd5e1', lineHeight: '1.4'}}>Inside local nodes, ThreadPools concurrently slice CPU power to process multiple NLP deep inferences universally simultaneously.</p>
          </div>
        </div>
      </div>
      
      </div>
    </>
  );
}

export default App;
