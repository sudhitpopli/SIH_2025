import React, { useState, useEffect, Component, ReactNode } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import Navbar from './navbar'; 
import About from './about';
import Features from './features';
import Home from './home';
import Contact from './contact';

import { useTrafficSocket } from '../hooks/useTrafficSocket';
import SimulationMap from './SimulationMap';

// Safe Error Boundary to prevent "White Screen of Death"
class DashboardErrorBoundary extends Component<{children: ReactNode}, {hasError: boolean, errorSnippet: string}> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false, errorSnippet: '' };
  }
  static getDerivedStateFromError(error: any) {
    return { hasError: true, errorSnippet: error.toString() };
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center h-[50vh] bg-red-950 text-red-500 font-mono p-10 rounded-3xl m-10 border border-red-500">
          <h2 className="text-3xl font-bold mb-4">CRITICAL FRONTEND CRASH</h2>
          <p>{this.state.errorSnippet}</p>
          <button onClick={() => window.location.reload()} className="mt-8 px-6 py-2 bg-red-800 text-white rounded">Hard Reboot UI</button>
        </div>
      );
    }
    return this.props.children;
  }
}

const TABS = ['Home', 'About Us', 'Features', 'Dashboard', 'Contact'];

// Reusable Theme Colors
const COLORS = {
  qmix: '#10b981', // Emerald
  standard: '#94a3b8', // Slate for native
  background: '#0f172a',
  grid: '#1e293b',
  text: '#64748b'
};

interface TimeSeriesData {
  time: number;
  standardReward: number;
  qmixReward: number;
  standardWait: number;
  qmixWait: number;
}

const ChartCard = ({ title, children }: { title: string, children: React.ReactNode }) => (
  <div className="bg-slate-800/40 backdrop-blur-md border border-slate-700/50 p-6 rounded-3xl shadow-xl h-[350px] flex flex-col w-full">
    <h2 className="text-xl font-bold text-slate-200 mb-4 text-center tracking-wide">{title}</h2>
    <div className="flex-grow w-full h-full">
      {children}
    </div>
  </div>
);

const StatsOverlay = ({ label, value, isPositive }: { label: string, value: string | number, isPositive?: boolean }) => (
  <div className="bg-slate-900/80 border border-slate-700/50 p-4 rounded-xl flex flex-col items-center justify-center flex-1 mx-2">
    <span className="text-sm text-slate-400 mb-1">{label}</span>
    <span className={`text-2xl font-black ${isPositive ? 'text-emerald-400' : 'text-slate-200'}`}>
      {value}
    </span>
  </div>
);

export default function TrafficDashboard() {
  const [activeTab, setActiveTab] = useState(() => {
    return localStorage.getItem('qmixActiveTab') || 'Home';
  });

  useEffect(() => {
    localStorage.setItem('qmixActiveTab', activeTab);
  }, [activeTab]);

  // Use the real backend WebSocket hook
  const { data, isConnected, isRunning, startSimulation, stopSimulation } = useTrafficSocket('ws://localhost:8000/ws/telemetry');
  
  // States for our charts and map
  const [timeSeriesData, setTimeSeriesData] = useState<TimeSeriesData[]>([]);
  const [mapData, setMapData] = useState<number[][][]>([]);

  // Fetch the real map layout once on boot
  useEffect(() => {
    fetch('http://localhost:8000/api/map')
      .then(res => res.json())
      .then(d => {
         if (d.roads) setMapData(d.roads);
      })
      .catch(e => console.error("Could not fetch map", e));
  }, []);

  // Update historical charts when new data comes in
  useEffect(() => {
    if (data && data.v2 && data.native) {
      setTimeSeriesData(prev => {
        const newData = [...prev, {
          time: data.native.step, // Use absolute backend timeline steps to avert generic Date.now() overlaps
          qmixReward: data.v2.reward,
          standardReward: data.native.reward,
          qmixWait: -data.v2.reward,
          standardWait: -data.native.reward
        }];
        // Keep last 30 points for smooth scrolling
        if (newData.length > 30) newData.shift();
        return newData;
      });

      // Auto-stop simulation strictly at Step 330 request
      if (isRunning && data.native.step >= 330) {
         stopSimulation();
      }
    }
  }, [data, isRunning, stopSimulation]);

  const formatTime = (label: any) => `Step ${label}`;

  // Extracted current metrics with deep optional chaining for safety
  const qmixCurrentReward = data?.v2?.reward || 0;
  const nativeCurrentReward = data?.native?.reward || 0;
  const qmixVehiclesCount = data?.v2?.vehicles?.length || 0;
  const nativeVehiclesCount = data?.native?.vehicles?.length || 0;

  return (
    <div className="min-h-screen bg-[#0B1121] text-slate-100 flex flex-col items-center pt-12 pb-10 px-6 font-sans selection:bg-emerald-500/30 overflow-x-hidden">
      
      <h1 className="text-6xl md:text-7xl font-black text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-400 tracking-tight mb-8 drop-shadow-sm text-center">
        QuMiks
      </h1>

      <Navbar tabs={TABS} activeTab={activeTab} onTabChange={setActiveTab} />

      <DashboardErrorBoundary>
        <div key={activeTab} className="animate-fade-up w-full flex flex-col items-center max-w-7xl">

        {activeTab === 'Home' ? (
          <Home onNavigate={setActiveTab} />
        ) : activeTab === 'Dashboard' ? (
          <>
            <div className="flex gap-4 mb-10">
              <button 
                onClick={isRunning ? stopSimulation : startSimulation}
                className={`px-8 py-3 rounded-xl font-bold tracking-widest text-sm uppercase transition-all duration-300 ${
                  isRunning 
                    ? 'bg-rose-500/10 text-rose-400 border border-rose-500/50 hover:bg-rose-500/20' 
                    : 'bg-emerald-500 text-slate-950 hover:bg-emerald-400 hover:-translate-y-0.5 shadow-[0_0_20px_rgba(16,185,129,0.3)]'
                }`}
              >
                {isRunning ? 'Halt Simulation' : 'Launch Engine'}
              </button>
              
              <div className={`px-4 py-3 rounded-xl border flex items-center gap-2 ${isConnected ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' : 'bg-rose-500/10 border-rose-500/30 text-rose-400'}`}>
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-emerald-500' : 'bg-rose-500'} animate-pulse`}></div>
                <span className="text-sm font-semibold">{isConnected ? 'BACKEND CONNECTED' : 'WAITING FOR API...'}</span>
              </div>
            </div>

            {/* Top Stat Bar */}
            <div className="flex w-full justify-between mb-8">
               <StatsOverlay label="AI Efficiency (Reward)" value={qmixCurrentReward.toFixed(2)} isPositive={qmixCurrentReward > nativeCurrentReward} />
               <StatsOverlay label="Native Efficiency (Reward)" value={nativeCurrentReward.toFixed(2)} />
               <StatsOverlay label="AI Vehicle Count" value={qmixVehiclesCount} />
               <StatsOverlay label="Native Vehicle Count" value={nativeVehiclesCount} />
            </div>

            {/* Simulation Maps (Side-by-Side) */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 w-full mb-8">
              <SimulationMap 
                title="Our Model ( Qmix Variant )" 
                vehicles={data?.v2?.vehicles || []} 
                mapRoads={mapData}
                isActive={true} 
              />
              <SimulationMap 
                title="SUMO Native (Standard Logic)" 
                vehicles={data?.native?.vehicles || []} 
                mapRoads={mapData}
                isActive={false} 
              />
            </div>

            {/* Performance Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 w-full">
              <ChartCard title="Live Efficiency Score (Reward)">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={timeSeriesData} margin={{ top: 5, right: 20, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={COLORS.grid} vertical={false} />
                    <XAxis 
                      dataKey="time" type="number" 
                      domain={['dataMin', 'dataMax']} 
                      tickFormatter={formatTime} stroke={COLORS.text} fontSize={11} minTickGap={30} 
                    />
                    <YAxis stroke={COLORS.text} fontSize={11} domain={['auto', 'auto']} />
                    <Tooltip labelFormatter={formatTime} contentStyle={{ backgroundColor: COLORS.background, borderColor: COLORS.grid, borderRadius: '8px' }} />
                    <Legend wrapperStyle={{ paddingTop: '10px' }} />
                    <Line type="monotone" dataKey="standardReward" name="Standard Model" stroke={COLORS.standard} strokeWidth={3} dot={false} isAnimationActive={false} />
                    <Line type="monotone" dataKey="qmixReward" name="Our Model ( Qmix Variant )" stroke={COLORS.qmix} strokeWidth={3} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </ChartCard>

              <ChartCard title="Cumulative Wait Time (Lower is Better)">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={timeSeriesData} margin={{ top: 5, right: 20, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={COLORS.grid} vertical={false} />
                    <XAxis 
                      dataKey="time" type="number" 
                      domain={['dataMin', 'dataMax']} 
                      tickFormatter={formatTime} stroke={COLORS.text} fontSize={11} minTickGap={30} 
                    />
                    <YAxis stroke={COLORS.text} fontSize={11} domain={['auto', 'auto']} />
                    <Tooltip labelFormatter={formatTime} contentStyle={{ backgroundColor: COLORS.background, borderColor: COLORS.grid, borderRadius: '8px' }} />
                    <Legend wrapperStyle={{ paddingTop: '10px' }} />
                    <Line type="monotone" dataKey="standardWait" name="Standard Model" stroke={COLORS.standard} strokeWidth={3} dot={false} isAnimationActive={false} />
                    <Line type="monotone" dataKey="qmixWait" name="Our Model ( Qmix Variant )" stroke={COLORS.qmix} strokeWidth={3} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </ChartCard>
            </div>
          </>
        ) : activeTab === 'About Us' ? (
          <About />
        ) :  activeTab === 'Features' ? (
          <Features />
        ) : activeTab === 'Contact' ? (
          <Contact />
        ) : (
          <div className="w-full max-w-6xl bg-slate-800/20 backdrop-blur-md border border-slate-700/50 p-12 rounded-3xl text-center">
            <h2 className="text-3xl font-bold text-slate-400">{activeTab} Content Area</h2>
            <p className="text-slate-500 mt-4">This section is currently under construction for the hackathon demo.</p>
          </div>
        )}

        </div>
      </DashboardErrorBoundary>
    </div>
  );
}