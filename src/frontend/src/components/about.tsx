import { useEffect, useRef, useState } from 'react';

// --- Reusable Scroll Reveal Component ---
const ScrollReveal = ({ children, delay = 0 }: { children: React.ReactNode, delay?: number }) => {
  const [isVisible, setIsVisible] = useState(false);
  const domRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          if (domRef.current) observer.unobserve(domRef.current);
        }
      });
    }, { threshold: 0.15 });

    if (domRef.current) observer.observe(domRef.current);
    return () => observer.disconnect();
  }, []);

  return (
    <div
      ref={domRef}
      className={`transition-all duration-1000 ease-out ${
        isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-24'
      }`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {children}
    </div>
  );
};

// --- The Upgraded Neural Visualizer Component ---
const NeuralVisualizer = () => {
  const [isLearning, setIsLearning] = useState(false);
  
  const [weights, setWeights] = useState(() => Array.from({ length: 28 }, () => Math.random() * 2 - 1));
  const [biases, setBiases] = useState(() => Array.from({ length: 8 }, () => Math.random() * 2 - 1));
  
  const [activeNodes, setActiveNodes] = useState<{ hidden: number[], out: number[] }>({ hidden: [], out: [] });
  const [hoveredNode, setHoveredNode] = useState<{layer: 'in'|'hid'|'out', index: number} | null>(null);

  const inputLabels = ["Queue Length", "Vehicle Density", "Phase Time"];
  const outputLabels = ["Extend Phase A", "Next Phase A->B", "Extend Phase B", "Emergency Halt"];

  useEffect(() => {
    if (!isLearning) {
      setActiveNodes({ hidden: [], out: [] });
      return;
    }

    const interval = setInterval(() => {
      setWeights(prev => prev.map(w => {
        const drift = (Math.random() * 0.1 - 0.05); 
        return Math.max(-1, Math.min(1, w + drift));
      }));

      setBiases(prev => prev.map(b => {
        const drift = (Math.random() * 0.06 - 0.03);
        return Math.max(-1, Math.min(1, b + drift));
      }));

      if (Math.random() > 0.3) {
        setActiveNodes({
          hidden: [Math.floor(Math.random() * 4)], 
          out: [Math.floor(Math.random() * 4)]
        });
      }
    }, 150); 

    return () => clearInterval(interval);
  }, [isLearning]);

  const isEdgeHighlighted = (fromLayer: 'in'|'hid', fromIdx: number, toIdx: number) => {
    if (!hoveredNode) return true; 
    if (hoveredNode.layer === 'in' && fromLayer === 'in' && hoveredNode.index === fromIdx) return true;
    if (hoveredNode.layer === 'hid' && (fromLayer === 'in' && hoveredNode.index === toIdx) || (fromLayer === 'hid' && hoveredNode.index === fromIdx)) return true;
    if (hoveredNode.layer === 'out' && fromLayer === 'hid' && hoveredNode.index === toIdx) return true;
    return false;
  };

  return (
    <div className="bg-slate-800/40 backdrop-blur-md border border-slate-700/50 p-6 md:p-10 rounded-3xl shadow-xl w-full relative overflow-hidden">
      <div className="absolute top-0 left-0 w-64 h-64 bg-cyan-500/10 rounded-full blur-[80px] pointer-events-none"></div>
      <div className="absolute bottom-0 right-0 w-64 h-64 bg-emerald-500/10 rounded-full blur-[80px] pointer-events-none"></div>

      <div className="flex flex-col items-center mb-8 relative z-10">
        <h3 className="text-3xl md:text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-400 tracking-wide mb-2 text-center">
          QMIX Neural Architect
        </h3>
        <p className="text-slate-400 text-sm mb-6 uppercase tracking-widest font-semibold">Hover over any node to inspect parameters</p>
        
        <button 
          onClick={() => setIsLearning(!isLearning)}
          className={`px-8 py-3 rounded-xl font-bold tracking-widest text-sm uppercase transition-all duration-300 ${
            isLearning 
              ? 'bg-rose-500/10 text-rose-400 border border-rose-500/50 hover:bg-rose-500/20' 
              : 'bg-emerald-500 text-slate-950 hover:bg-emerald-400 hover:-translate-y-0.5 shadow-[0_0_20px_rgba(16,185,129,0.3)]'
          }`}
        >
          {isLearning ? 'Halt Training Feed' : 'Initiate Live Training'}
        </button>
      </div>

      <div className="w-full overflow-x-auto relative z-10">
        <div className="min-w-[900px]"> 
          <svg viewBox="0 0 1000 650" className="w-full h-auto">
            
            {/* 1. Draw Edges: Input to Hidden */}
            {inputLabels.map((_, i) => 
              Array.from({ length: 4 }).map((_, j) => {
                const w = weights[i * 4 + j];
                // THE FIX: Changed negative weights to Slate Grey (#64748b)
                const color = w > 0 ? '#10b981' : '#64748b';
                const isHovered = isEdgeHighlighted('in', i, j);
                
                let opacity = 0;
                if (hoveredNode) opacity = isHovered ? 1 : 0.02;
                else opacity = Math.abs(w) > 0.5 ? Math.abs(w) * 0.6 : 0.15; // Bumped base opacity slightly for the grey

                const width = hoveredNode && isHovered ? 3 : (Math.abs(w) * 3 + 0.5);
                const x1 = 250, y1 = 160 + i * 160, x2 = 500, y2 = 140 + j * 120;

                return (
                  <g key={`edge-in-${i}-${j}`}>
                    <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={width} strokeOpacity={opacity} className="transition-all duration-150 ease-linear" />
                    {hoveredNode && isHovered && (
                      <g transform={`translate(${(x1+x2)/2}, ${(y1+y2)/2})`}>
                        <rect x="-20" y="-10" width="40" height="20" rx="4" fill="#0f172a" stroke={color} strokeWidth="1" opacity="0.9" />
                        <text x="0" y="3" textAnchor="middle" fill="#f8fafc" fontSize="10" fontWeight="bold" fontFamily="monospace">
                          {w > 0 ? '+' : ''}{w.toFixed(2)}
                        </text>
                      </g>
                    )}
                  </g>
                );
              })
            )}

            {/* 2. Draw Edges: Hidden to Output */}
            {Array.from({ length: 4 }).map((_, j) => 
              outputLabels.map((_, k) => {
                const w = weights[12 + j * 4 + k];
                // THE FIX: Changed negative weights to Slate Grey (#64748b)
                const color = w > 0 ? '#10b981' : '#64748b';
                const isHovered = isEdgeHighlighted('hid', j, k);
                
                let opacity = 0;
                if (hoveredNode) opacity = isHovered ? 1 : 0.02;
                else opacity = Math.abs(w) > 0.5 ? Math.abs(w) * 0.6 : 0.15;

                const width = hoveredNode && isHovered ? 3 : (Math.abs(w) * 3 + 0.5);
                const x1 = 500, y1 = 140 + j * 120, x2 = 750, y2 = 140 + k * 120;

                return (
                  <g key={`edge-out-${j}-${k}`}>
                    <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={width} strokeOpacity={opacity} className="transition-all duration-150 ease-linear" />
                    {hoveredNode && isHovered && (
                      <g transform={`translate(${(x1+x2)/2}, ${(y1+y2)/2})`}>
                        <rect x="-20" y="-10" width="40" height="20" rx="4" fill="#0f172a" stroke={color} strokeWidth="1" opacity="0.9" />
                        <text x="0" y="3" textAnchor="middle" fill="#f8fafc" fontSize="10" fontWeight="bold" fontFamily="monospace">
                          {w > 0 ? '+' : ''}{w.toFixed(2)}
                        </text>
                      </g>
                    )}
                  </g>
                );
              })
            )}

            {/* 3. Draw Nodes: Input Layer */}
            {inputLabels.map((label, i) => (
              <g 
                key={`node-in-${i}`} 
                onMouseEnter={() => setHoveredNode({layer: 'in', index: i})}
                onMouseLeave={() => setHoveredNode(null)}
                className="cursor-pointer"
              >
                {/* Text pushed back slightly to account for bigger node */}
                <text x={215} y={160 + i * 160} textAnchor="end" alignmentBaseline="middle" fill={hoveredNode?.layer === 'in' && hoveredNode.index === i ? '#38bdf8' : '#cbd5e1'} fontSize={14} fontWeight="bold" className="transition-colors">{label}</text>
                {/* THE FIX: Bigger radius (r=22) and color changed to Emerald (#10b981) */}
                <circle cx={250} cy={160 + i * 160} r={22} fill="#10b981" stroke={hoveredNode?.layer === 'in' && hoveredNode.index === i ? '#ffffff' : '#0f172a'} strokeWidth={2} className="transition-all drop-shadow-md" />
              </g>
            ))}

            {/* 4. Draw Nodes & Biases: Hidden Layer */}
            {Array.from({ length: 4 }).map((_, j) => (
              <g 
                key={`node-hid-${j}`}
                onMouseEnter={() => setHoveredNode({layer: 'hid', index: j})}
                onMouseLeave={() => setHoveredNode(null)}
                className="cursor-pointer"
              >
                {/* THE FIX: Scaled up pulse (r=32) and base node (r=22) */}
                {isLearning && activeNodes.hidden.includes(j) && <circle cx={500} cy={140 + j * 120} r={32} fill="#94a3b8" opacity={0.4} className="animate-pulse" />}
                <circle cx={500} cy={140 + j * 120} r={22} fill="#64748b" stroke={hoveredNode?.layer === 'hid' && hoveredNode.index === j ? '#ffffff' : '#0f172a'} strokeWidth={2} className="transition-all drop-shadow-md" />
                
                {/* BIAS TEXT: Pushed down slightly for bigger node */}
                <text x={500} y={140 + j * 120 + 38} textAnchor="middle" fill="#94a3b8" fontSize={12} fontFamily="monospace" fontWeight="bold">
                  b: {biases[j] > 0 ? '+' : ''}{biases[j].toFixed(2)}
                </text>
              </g>
            ))}

            {/* 5. Draw Nodes & Biases: Output Layer */}
            {outputLabels.map((label, k) => (
              <g 
                key={`node-out-${k}`}
                onMouseEnter={() => setHoveredNode({layer: 'out', index: k})}
                onMouseLeave={() => setHoveredNode(null)}
                className="cursor-pointer"
              >
                {/* THE FIX: Scaled up pulse (r=32) and base node (r=22) */}
                {isLearning && activeNodes.out.includes(k) && <circle cx={750} cy={140 + k * 120} r={32} fill="#10b981" opacity={0.4} className="animate-pulse" />}
                <circle cx={750} cy={140 + k * 120} r={22} fill="#10b981" stroke={hoveredNode?.layer === 'out' && hoveredNode.index === k ? '#ffffff' : '#0f172a'} strokeWidth={2} className="transition-all drop-shadow-md" />
                {/* Text pushed forward slightly to account for bigger node */}
                <text x={785} y={140 + k * 120} textAnchor="start" alignmentBaseline="middle" fill={hoveredNode?.layer === 'out' && hoveredNode.index === k ? '#10b981' : '#f8fafc'} fontSize={15} fontWeight="bold" className="transition-colors">{label}</text>
                
                {/* BIAS TEXT: Pushed down slightly for bigger node */}
                <text x={750} y={140 + k * 120 + 38} textAnchor="middle" fill="#94a3b8" fontSize={12} fontFamily="monospace" fontWeight="bold">
                  b: {biases[4 + k] > 0 ? '+' : ''}{biases[4 + k].toFixed(2)}
                </text>
              </g>
            ))}

            {/* Layer Titles */}
            <text x={250} y={30} textAnchor="middle" fill="#cbd5e1" fontSize={16} fontWeight="bold" className="uppercase tracking-widest">Input State</text>
            <text x={500} y={30} textAnchor="middle" fill="#cbd5e1" fontSize={16} fontWeight="bold" className="uppercase tracking-widest">Hidden Processing</text>
            <text x={750} y={30} textAnchor="middle" fill="#cbd5e1" fontSize={16} fontWeight="bold" className="uppercase tracking-widest">Action Logits</text>
          </svg>
        </div>
      </div>
    </div>
  );
};

export default function About() {
  return (
    <div className="w-full max-w-7xl mx-auto py-12 flex flex-col gap-32">
      <ScrollReveal>
        <section className="text-center flex flex-col items-center">
          <h2 className="text-5xl md:text-6xl font-black text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-400 tracking-tight mb-10">
            About Our Project
          </h2>
          <p className="text-slate-300 text-xl leading-relaxed max-w-5xl">
            We are pioneering the future of urban mobility with an AI-driven traffic management system designed to optimize signal timings and reduce congestion in cities. Leveraging real-time data from cameras and IoT sensors, our platform uses cutting-edge computer vision and reinforcement learning technologies to predict bottlenecks and adapt traffic flow dynamically.
          </p>
        </section>
      </ScrollReveal>

      <ScrollReveal>
        <NeuralVisualizer />
      </ScrollReveal>

      <ScrollReveal>
        <section className="grid grid-cols-1 md:grid-cols-2 gap-12 md:gap-24 items-center">
          <div className="flex flex-col items-start text-left">
            <h3 className="text-4xl font-bold text-slate-200 mb-6">Our Mission</h3>
            <p className="text-slate-400 text-xl leading-relaxed">
              To develop an AI-driven traffic management system that leverages real-time data from cameras and IoT sensors to optimize traffic signals, reduce congestion, and improve the daily commuting experience in urban areas.
            </p>
          </div>
          <div className="hidden md:flex justify-center items-center relative w-full h-80">
            <div className="absolute inset-0 bg-gradient-to-br from-slate-800/50 to-slate-900/50 rounded-[2.5rem] border border-slate-700/50 overflow-hidden">
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-48 bg-emerald-500/20 rounded-full blur-[60px]"></div>
              <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:2rem_2rem] opacity-30"></div>
            </div>
          </div>
        </section>
      </ScrollReveal>

      <ScrollReveal>
        <section className="grid grid-cols-1 md:grid-cols-2 gap-12 md:gap-24 items-center pb-20">
          <div className="hidden md:flex justify-center items-center relative w-full h-80 order-last md:order-first">
            <div className="absolute inset-0 bg-gradient-to-bl from-slate-800/50 to-slate-900/50 rounded-[2.5rem] border border-slate-700/50 overflow-hidden">
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-48 bg-cyan-500/20 rounded-full blur-[60px]"></div>
              <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:2rem_2rem] opacity-30"></div>
            </div>
          </div>
          <div className="flex flex-col items-start md:items-end text-left md:text-right">
            <h3 className="text-4xl font-bold text-slate-200 mb-6">Our Vision</h3>
            <p className="text-slate-400 text-xl leading-relaxed">
              To build smarter, safer, and more sustainable cities by transforming traditional traffic control into an intelligent, adaptive ecosystem that minimizes travel time, lowers emissions, and enhances quality of life for all citizens.
            </p>
          </div>
        </section>
      </ScrollReveal>
    </div>
  );
}