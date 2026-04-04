import { useState, useEffect } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell
} from 'recharts';
import Navbar from './navbar'; 
import About from './about';
import Features from './features';
import Home from './home';
import Contact from './contact';

const TABS = ['Home', 'About Us', 'Features', 'Dashboard', 'Simulation', 'Contact'];
const TICK_RATE_MS = 100;

// Reusable Theme Colors
const COLORS = {
  qmix: '#10b981', // Emerald
  standard: '#f43f5e', // Rose
  background: '#0f172a',
  grid: '#1e293b',
  text: '#64748b'
};

interface TimeSeriesData {
  time: number;
  standardWait: number;
  qmixWait: number;
  standardReward: number;
  qmixReward: number;
}

// Reusable Chart Container Component
const ChartCard = ({ title, children }: { title: string, children: React.ReactNode }) => (
  <div className="bg-slate-800/40 backdrop-blur-md border border-slate-700/50 p-6 rounded-3xl shadow-xl h-[400px] flex flex-col w-full">
    <h2 className="text-xl font-bold text-slate-200 mb-4 text-center tracking-wide">{title}</h2>
    <div className="flex-grow w-full h-full">
      {children}
    </div>
  </div>
);

export default function TrafficDashboard() {
  const [activeTab, setActiveTab] = useState(() => {
    const savedTab = localStorage.getItem('qmixActiveTab');
    return savedTab || 'Home';
  });

  // 2. Every time activeTab changes, save the new value to local storage
  useEffect(() => {
    localStorage.setItem('qmixActiveTab', activeTab);
  }, [activeTab]);
  const [isRunning, setIsRunning] = useState(false);
  
  // States for our different charts
  const [timeSeriesData, setTimeSeriesData] = useState<TimeSeriesData[]>([]);
  const [pieData, setPieData] = useState([
    { name: 'QMIX Processed', value: 50 },
    { name: 'Standard Processed', value: 50 }
  ]);
  const [barData, setBarData] = useState([
    { metric: 'Avg Wait (s)', Standard: 65, QMIX: 25 },
    { metric: 'Max Queue', Standard: 22, QMIX: 6 },
    { metric: 'Throughput', Standard: 140, QMIX: 140 }
  ]);

  // High-Frequency Data Engine
  useEffect(() => {
    if (!isRunning) return;

    let tickCount = 0;

    const interval = setInterval(() => {
      tickCount++;

      // 1. Update Line Charts (Runs every 100ms for smooth flow)
      setTimeSeriesData((prevData) => {
        const maxPoints = Math.ceil(60000 / TICK_RATE_MS);
        const newData = [...prevData].slice(-maxPoints);
        
        const last = prevData.length > 0 ? prevData[prevData.length - 1] : null;
        
        const currentPoint: TimeSeriesData = {
          time: Date.now(),
          // Wait Times
          standardWait: Math.max(0, (last?.standardWait || 42) + (Math.random() * 0.15 + 0.05)),
          qmixWait: Math.max(0, (last?.qmixWait || 25) + (Math.random() * 0.04 - 0.02)),
          // Rewards (Standard stays negative/flat, QMIX learns and climbs)
          standardReward: (last?.standardReward || -40) + (Math.random() * 0.2 - 0.1),
          qmixReward: Math.min(95, (last?.qmixReward || -20) + (Math.random() * 0.5 + 0.05)), 
        };

        return [...newData, currentPoint];
      });

      // 2. Update Pie and Bar Charts (Runs slower, every 1 second, so they don't visually jitter)
      if (tickCount % 10 === 0) {
        setPieData(prev => [
          { name: 'QMIX Processed', value: prev[0].value + Math.floor(Math.random() * 8 + 4) },
          { name: 'Standard Processed', value: prev[1].value + Math.floor(Math.random() * 5 + 1) }
        ]);

        setBarData(prev => [
          { ...prev[0], Standard: prev[0].Standard + (Math.random() * 0.4), QMIX: 25 + (Math.random() * 1 - 0.5) },
          { ...prev[1], Standard: prev[1].Standard + (Math.random() * 0.2), QMIX: 6 + (Math.random() * 0.5) },
          { ...prev[2], Standard: prev[2].Standard + 2, QMIX: prev[2].QMIX + 6 }
        ]);
      }

    }, TICK_RATE_MS); 

    return () => clearInterval(interval);
  }, [isRunning]);

  const formatTime = (label: any) => new Date(Number(label)).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

  return (
    <div className="min-h-screen bg-[#0B1121] text-slate-100 flex flex-col items-center pt-12 pb-10 px-6 font-sans selection:bg-emerald-500/30 overflow-x-hidden">
      
      {/* Main Title */}
      <h1 className="text-6xl md:text-7xl font-black text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-400 tracking-tight mb-8 drop-shadow-sm text-center">
        QuMiks
      </h1>

      {/* Interactive Navbar Component */}
      <Navbar tabs={TABS} activeTab={activeTab} onTabChange={setActiveTab} />

      {/* Dynamic Content Router with Fade-Up Animation */}
      <div key={activeTab} className="animate-fade-up w-full flex flex-col items-center max-w-7xl">

        {activeTab === 'Home' ? (
          /* --- The New Home Route --- */
          <Home onNavigate={setActiveTab} />
        ) : activeTab === 'Dashboard' ? (
          <>
            <button 
              onClick={() => setIsRunning(!isRunning)}
              className={`mb-10 px-8 py-3 rounded-xl font-bold tracking-widest text-sm uppercase transition-all duration-300 ${
                isRunning 
                  ? 'bg-rose-500/10 text-rose-400 border border-rose-500/50 hover:bg-rose-500/20' 
                  : 'bg-emerald-500 text-slate-950 hover:bg-emerald-400 hover:-translate-y-0.5 shadow-[0_0_20px_rgba(16,185,129,0.3)]'
              }`}
            >
              {isRunning ? 'Halt Live Feed' : 'Initiate Live Feed'}
            </button>

            {/* --- The 2x2 Responsive Grid --- */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 w-full">
              
              {/* 1. Wait Time Line Chart */}
              <ChartCard title="Intersection Wait Time">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={timeSeriesData} margin={{ top: 5, right: 20, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={COLORS.grid} vertical={false} />
                    <XAxis 
                      dataKey="time" type="number" 
                      domain={[ timeSeriesData.length > 0 ? timeSeriesData[0].time : 'dataMin', timeSeriesData.length > 0 ? timeSeriesData[0].time + 60000 : 'dataMax' ]} 
                      tickFormatter={formatTime} stroke={COLORS.text} fontSize={11} minTickGap={30} 
                    />
                    <YAxis stroke={COLORS.text} fontSize={11} tickFormatter={(v) => `${v}s`} domain={[0, 80]} />
                    <Tooltip labelFormatter={formatTime} contentStyle={{ backgroundColor: COLORS.background, borderColor: COLORS.grid, borderRadius: '8px' }} />
                    <Legend wrapperStyle={{ paddingTop: '10px' }} />
                    <Line type="monotone" dataKey="standardWait" name="Standard Model" stroke={COLORS.standard} strokeWidth={3} dot={false} isAnimationActive={false} />
                    <Line type="monotone" dataKey="qmixWait" name="Our QMIX Model" stroke={COLORS.qmix} strokeWidth={3} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </ChartCard>

              {/* 2. RL Reward Line Chart */}
              <ChartCard title="Average Agent Reward">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={timeSeriesData} margin={{ top: 5, right: 20, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={COLORS.grid} vertical={false} />
                    <XAxis 
                      dataKey="time" type="number" 
                      domain={[ timeSeriesData.length > 0 ? timeSeriesData[0].time : 'dataMin', timeSeriesData.length > 0 ? timeSeriesData[0].time + 60000 : 'dataMax' ]} 
                      tickFormatter={formatTime} stroke={COLORS.text} fontSize={11} minTickGap={30} 
                    />
                    <YAxis stroke={COLORS.text} fontSize={11} domain={[-60, 100]} />
                    <Tooltip labelFormatter={formatTime} contentStyle={{ backgroundColor: COLORS.background, borderColor: COLORS.grid, borderRadius: '8px' }} />
                    <Legend wrapperStyle={{ paddingTop: '10px' }} />
                    <Line type="monotone" dataKey="standardReward" name="Standard Model" stroke={COLORS.standard} strokeWidth={3} dot={false} isAnimationActive={false} />
                    <Line type="monotone" dataKey="qmixReward" name="Our QMIX Model" stroke={COLORS.qmix} strokeWidth={3} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </ChartCard>

              {/* 3. Performance Bar Chart */}
              <ChartCard title="Performance Metrics">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={barData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={COLORS.grid} vertical={false} />
                    <XAxis dataKey="metric" stroke={COLORS.text} fontSize={12} />
                    <YAxis stroke={COLORS.text} fontSize={11} />
                    <Tooltip cursor={{ fill: 'rgba(255,255,255,0.05)' }} contentStyle={{ backgroundColor: COLORS.background, borderColor: COLORS.grid, borderRadius: '8px' }} />
                    <Legend wrapperStyle={{ paddingTop: '10px' }} />
                    <Bar dataKey="Standard" name="Standard Model" fill={COLORS.standard} radius={[4, 4, 0, 0]} />
                    <Bar dataKey="QMIX" name="Our QMIX Model" fill={COLORS.qmix} radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>

              {/* 4. Vehicle Distribution Pie Chart */}
              <ChartCard title="Total Vehicles Cleared">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={pieData}
                      cx="50%" cy="45%"
                      innerRadius={80}
                      outerRadius={110}
                      paddingAngle={5}
                      dataKey="value"
                      stroke="none"
                    >
                      {pieData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={index === 0 ? COLORS.qmix : COLORS.standard} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ backgroundColor: COLORS.background, borderColor: COLORS.grid, borderRadius: '8px' }} />
                    <Legend verticalAlign="bottom" height={36} />
                  </PieChart>
                </ResponsiveContainer>
              </ChartCard>

            </div>
          </>
        ) : activeTab === 'About Us' ? (
          /* --- The New About Us Route --- */
          <About />
        ) :  activeTab === 'Features' ? (
          /* --- The New Features Route --- */
          <Features />
        ) : activeTab === 'Contact' ? (
          /* --- The New Contact Route --- */
          <Contact />
        ) : (
          /* --- Fallback for all other tabs --- */
          <div className="w-full max-w-6xl bg-slate-800/20 backdrop-blur-md border border-slate-700/50 p-12 rounded-3xl text-center">
            <h2 className="text-3xl font-bold text-slate-400">{activeTab} Content Area</h2>
            <p className="text-slate-500 mt-4">This section is currently under construction for the hackathon demo.</p>
          </div>
        )}

      </div>
    </div>
  );
}