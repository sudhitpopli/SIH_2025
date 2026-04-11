import { useEffect, useRef, useState, type ReactNode } from 'react';

// --- Reusable Scroll Reveal Component ---
const ScrollReveal = ({ children, delay = 0 }: { children: ReactNode, delay?: number }) => {
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
    className={`transition-all duration-1000 ease-out ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-24'
      }`}
    style={{ transitionDelay: `${delay}ms` }}
  >
    {children}
  </div>
  );
};

// --- Shared Constants & Helpers ---
const INPUT_LABELS = ["Queue Length", "Vehicle Density", "Phase Time"];
const OUTPUT_LABELS = ["Extend Phase A", "Next Phase A->B", "Extend Phase B", "Emergency Halt"];
const POS_COLOR = "#10b981"; // Emerald
const NEG_COLOR = "#64748b"; // Slate Grey

// --- 1. Feedforward ANN Visualizer (Used for System A & B) ---
const FeedforwardNetwork = ({ title, desc, isLearning, seed }: { title: string, desc: string, isLearning: boolean, seed: number }) => {
  const [weights, setWeights] = useState(() => Array.from({ length: 28 }, (_, i) => Math.sin(seed * (i + 1)) * 2 - 1));
  const [biases, setBiases] = useState(() => Array.from({ length: 8 }, (_, i) => Math.cos(seed * (i + 1)) * 2 - 1));

  const [activeNodes, setActiveNodes] = useState<{ hidden: number[], out: number[] }>({ hidden: [], out: [] });
  const [hoveredNode, setHoveredNode] = useState<{ layer: 'in' | 'hid' | 'out', index: number } | null>(null);

  useEffect(() => {
    if (!isLearning) return;
    const interval = setInterval(() => {
      setWeights(prev => prev.map(w => Math.max(-1, Math.min(1, w + (Math.random() * 0.1 - 0.05)))));
      setBiases(prev => prev.map(b => Math.max(-1, Math.min(1, b + (Math.random() * 0.06 - 0.03)))));
      if (Math.random() > 0.3) {
        setActiveNodes({ hidden: [Math.floor(Math.random() * 4)], out: [Math.floor(Math.random() * 4)] });
      }
    }, 150);
    return () => clearInterval(interval);
  }, [isLearning]);

  const isEdgeHighlighted = (fromLayer: 'in' | 'hid', fromIdx: number, toIdx: number) => {
    if (!hoveredNode) return true;
    if (hoveredNode.layer === 'in' && fromLayer === 'in' && hoveredNode.index === fromIdx) return true;
    if (hoveredNode.layer === 'hid' && ((fromLayer === 'in' && hoveredNode.index === toIdx) || (fromLayer === 'hid' && hoveredNode.index === fromIdx))) return true;
    if (hoveredNode.layer === 'out' && fromLayer === 'hid' && hoveredNode.index === toIdx) return true;
    return false;
  };

  return (
    <div className="bg-slate-800/40 backdrop-blur-md border border-slate-700/50 p-6 md:p-8 rounded-3xl shadow-xl w-full relative overflow-hidden mb-12">
      <div className="absolute top-0 right-0 w-64 h-64 bg-emerald-500/5 rounded-full blur-[80px] pointer-events-none"></div>

      <div className="mb-6">
        <h4 className="text-2xl font-bold text-slate-200">{title}</h4>
        <p className="text-slate-400 text-sm mt-1">{desc}</p>
      </div>

      <div className="w-full overflow-x-auto relative z-10">
        <div className="min-w-[900px]">
          <svg viewBox="0 0 1000 600" className="w-full h-auto">
            {/* Edges: Input -> Hidden */}
            {INPUT_LABELS.map((_, i) => Array.from({ length: 4 }).map((_, j) => {
              const w = weights[i * 4 + j];
              const color = w > 0 ? POS_COLOR : NEG_COLOR;
              const isHovered = isEdgeHighlighted('in', i, j);
              const opacity = hoveredNode ? (isHovered ? 1 : 0.02) : (Math.abs(w) > 0.5 ? Math.abs(w) * 0.5 : 0.15);
              // THE FIX: Reduced base thickness multiplier from *3 to *1.5
              const width = hoveredNode && isHovered ? 2.5 : (Math.abs(w) * 1.5 + 0.25);
              const x1 = 250, y1 = 155 + i * 145, x2 = 500, y2 = 110 + j * 125;
              return (
                <g key={`edge-in-${i}-${j}`}>
                  <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={width} strokeOpacity={opacity} className="transition-all duration-150 ease-linear" />
                  {hoveredNode && isHovered && (
                    <g transform={`translate(${(x1 + x2) / 2}, ${(y1 + y2) / 2})`}>
                      <rect x="-20" y="-10" width="40" height="20" rx="4" fill="#0f172a" stroke={color} strokeWidth="1" opacity="0.9" />
                      <text x="0" y="3" textAnchor="middle" fill="#f8fafc" fontSize="10" fontWeight="bold" fontFamily="monospace">{w > 0 ? '+' : ''}{w.toFixed(2)}</text>
                    </g>
                  )}
                </g>
              );
            }))}

            {/* Edges: Hidden -> Output */}
            {Array.from({ length: 4 }).map((_, j) => OUTPUT_LABELS.map((_, k) => {
              const w = weights[12 + j * 4 + k];
              const color = w > 0 ? POS_COLOR : NEG_COLOR;
              const isHovered = isEdgeHighlighted('hid', j, k);
              const opacity = hoveredNode ? (isHovered ? 1 : 0.02) : (Math.abs(w) > 0.5 ? Math.abs(w) * 0.5 : 0.15);
              // THE FIX: Reduced base thickness multiplier from *3 to *1.5
              const width = hoveredNode && isHovered ? 2.5 : (Math.abs(w) * 1.5 + 0.25);
              const x1 = 500, y1 = 110 + j * 125, x2 = 750, y2 = 110 + k * 125;
              return (
                <g key={`edge-out-${j}-${k}`}>
                  <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={width} strokeOpacity={opacity} className="transition-all duration-150 ease-linear" />
                  {hoveredNode && isHovered && (
                    <g transform={`translate(${(x1 + x2) / 2}, ${(y1 + y2) / 2})`}>
                      <rect x="-20" y="-10" width="40" height="20" rx="4" fill="#0f172a" stroke={color} strokeWidth="1" opacity="0.9" />
                      <text x="0" y="3" textAnchor="middle" fill="#f8fafc" fontSize="10" fontWeight="bold" fontFamily="monospace">{w > 0 ? '+' : ''}{w.toFixed(2)}</text>
                    </g>
                  )}
                </g>
              );
            }))}

            {/* Input Nodes (Emerald) */}
            {INPUT_LABELS.map((label, i) => (
              <g key={`in-${i}`} onMouseEnter={() => setHoveredNode({ layer: 'in', index: i })} onMouseLeave={() => setHoveredNode(null)} className="cursor-pointer">
                <text x={215} y={155 + i * 145} textAnchor="end" alignmentBaseline="middle" fill={hoveredNode?.layer === 'in' && hoveredNode.index === i ? '#10b981' : '#cbd5e1'} fontSize={14} fontWeight="bold" className="transition-colors">{label}</text>
                <circle cx={250} cy={155 + i * 145} r={22} fill="#10b981" stroke={hoveredNode?.layer === 'in' && hoveredNode.index === i ? '#ffffff' : '#0f172a'} strokeWidth={2} className="transition-all drop-shadow-md" />
              </g>
            ))}

            {/* Hidden Nodes (Slate Grey) */}
            {Array.from({ length: 4 }).map((_, j) => (
              <g key={`hid-${j}`} onMouseEnter={() => setHoveredNode({ layer: 'hid', index: j })} onMouseLeave={() => setHoveredNode(null)} className="cursor-pointer">
                {isLearning && activeNodes.hidden.includes(j) && <circle cx={500} cy={110 + j * 125} r={32} fill="#94a3b8" opacity={0.4} className="animate-pulse" />}
                <circle cx={500} cy={110 + j * 125} r={22} fill="#64748b" stroke={hoveredNode?.layer === 'hid' && hoveredNode.index === j ? '#ffffff' : '#0f172a'} strokeWidth={2} className="transition-all drop-shadow-md" />
                <text x={500} y={110 + j * 125 + 38} textAnchor="middle" fill="#94a3b8" fontSize={12} fontFamily="monospace" fontWeight="bold">b: {biases[j] > 0 ? '+' : ''}{biases[j].toFixed(2)}</text>
              </g>
            ))}

            {/* Output Nodes (Emerald) */}
            {OUTPUT_LABELS.map((label, k) => (
              <g key={`out-${k}`} onMouseEnter={() => setHoveredNode({ layer: 'out', index: k })} onMouseLeave={() => setHoveredNode(null)} className="cursor-pointer">
                {isLearning && activeNodes.out.includes(k) && <circle cx={750} cy={110 + k * 125} r={32} fill="#10b981" opacity={0.4} className="animate-pulse" />}
                <circle cx={750} cy={110 + k * 125} r={22} fill="#10b981" stroke={hoveredNode?.layer === 'out' && hoveredNode.index === k ? '#ffffff' : '#0f172a'} strokeWidth={2} className="transition-all drop-shadow-md" />
                <text x={785} y={110 + k * 125} textAnchor="start" alignmentBaseline="middle" fill={hoveredNode?.layer === 'out' && hoveredNode.index === k ? '#10b981' : '#f8fafc'} fontSize={15} fontWeight="bold" className="transition-colors">{label}</text>
                <text x={750} y={110 + k * 125 + 38} textAnchor="middle" fill="#94a3b8" fontSize={12} fontFamily="monospace" fontWeight="bold">b: {biases[4 + k] > 0 ? '+' : ''}{biases[4 + k].toFixed(2)}</text>
              </g>
            ))}
          </svg>
        </div>
      </div>
    </div>
  );
};

// --- 2. Recurrent Neural Network Visualizer (System C) ---
const RecurrentNetwork = ({ title, desc, isLearning }: { title: string, desc: string, isLearning: boolean }) => {
  const [weights, setWeights] = useState(() => Array.from({ length: 44 }, (_, i) => Math.cos(i * 1.5) * 2 - 1));
  const [recWeights, setRecWeights] = useState(() => Array.from({ length: 8 }, (_, i) => Math.sin(i * 3) * 2 - 1));
  const [biases, setBiases] = useState(() => Array.from({ length: 12 }, (_, i) => Math.sin(i) * 2 - 1));

  const [activeNodes, setActiveNodes] = useState<{ h1: number[], h2: number[], out: number[] }>({ h1: [], h2: [], out: [] });
  const [hoveredNode, setHoveredNode] = useState<{ layer: 'in' | 'h1' | 'h2' | 'out', index: number } | null>(null);

  useEffect(() => {
    if (!isLearning) return;
    const interval = setInterval(() => {
      setWeights(prev => prev.map(w => Math.max(-1, Math.min(1, w + (Math.random() * 0.1 - 0.05)))));
      setRecWeights(prev => prev.map(w => Math.max(-1, Math.min(1, w + (Math.random() * 0.1 - 0.05)))));
      setBiases(prev => prev.map(b => Math.max(-1, Math.min(1, b + (Math.random() * 0.06 - 0.03)))));
      if (Math.random() > 0.3) {
        setActiveNodes({ h1: [Math.floor(Math.random() * 4)], h2: [Math.floor(Math.random() * 4)], out: [Math.floor(Math.random() * 4)] });
      }
    }, 150);
    return () => clearInterval(interval);
  }, [isLearning]);

  const isEdgeHighlighted = (fromLayer: 'in' | 'h1' | 'h2', fromIdx: number, toIdx: number) => {
    if (!hoveredNode) return true;
    if (hoveredNode.layer === 'in' && fromLayer === 'in' && hoveredNode.index === fromIdx) return true;
    if (hoveredNode.layer === 'h1' && ((fromLayer === 'in' && hoveredNode.index === toIdx) || (fromLayer === 'h1' && hoveredNode.index === fromIdx))) return true;
    if (hoveredNode.layer === 'h2' && ((fromLayer === 'h1' && hoveredNode.index === toIdx) || (fromLayer === 'h2' && hoveredNode.index === fromIdx))) return true;
    if (hoveredNode.layer === 'out' && fromLayer === 'h2' && hoveredNode.index === toIdx) return true;
    return false;
  };

  const isLoopHighlighted = (layer: 'h1' | 'h2', idx: number) => {
    if (!hoveredNode) return true;
    return hoveredNode.layer === layer && hoveredNode.index === idx;
  };

  const drawEdges = (startX: number, endX: number, startLayer: 'in' | 'h1' | 'h2', weightOffset: number, fromCount: number, toCount: number) => {
    return Array.from({ length: fromCount }).map((_, j) => Array.from({ length: toCount }).map((_, k) => {
      const w = weights[weightOffset + j * toCount + k];
      const color = w > 0 ? POS_COLOR : NEG_COLOR;
      const isHovered = isEdgeHighlighted(startLayer, j, k);
      const opacity = hoveredNode ? (isHovered ? 1 : 0.02) : (Math.abs(w) > 0.5 ? Math.abs(w) * 0.5 : 0.15);
      // THE FIX: Reduced base thickness multiplier from *3 to *1.5
      const width = hoveredNode && isHovered ? 2.5 : (Math.abs(w) * 1.5 + 0.25);
      const y1 = startLayer === 'in' ? 155 + j * 145 : 110 + j * 125;
      const y2 = 110 + k * 125;
      return (
        <g key={`edge-${startLayer}-${j}-${k}`}>
          <line x1={startX} y1={y1} x2={endX} y2={y2} stroke={color} strokeWidth={width} strokeOpacity={opacity} className="transition-all duration-150 ease-linear" />
        </g>
      );
    }));
  };

  const drawRecurrentLoops = (cx: number, layer: 'h1' | 'h2', weightOffset: number) => {
    return Array.from({ length: 4 }).map((_, j) => {
      const w = recWeights[weightOffset + j];
      const color = w > 0 ? '#38bdf8' : '#818cf8';
      const isHovered = isLoopHighlighted(layer, j);
      const opacity = hoveredNode ? (isHovered ? 1 : 0.05) : (Math.abs(w) > 0.5 ? Math.abs(w) * 0.8 : 0.3);
      // THE FIX: Reduced base loop thickness multiplier
      const width = hoveredNode && isHovered ? 3 : (Math.abs(w) * 1.5 + 0.75);
      const cy = 110 + j * 125;
      const pathData = `M ${cx - 8} ${cy - 20} C ${cx - 40} ${cy - 70}, ${cx + 40} ${cy - 70}, ${cx + 8} ${cy - 20}`;

      return (
        <g key={`loop-${layer}-${j}`}>
          <path d={pathData} fill="none" stroke={color} strokeWidth={width} strokeOpacity={opacity} className="transition-all duration-150 ease-linear" strokeDasharray="4 2" />
          {hoveredNode && isHovered && (
            <g transform={`translate(${cx}, ${cy - 60})`}>
              <rect x="-20" y="-10" width="40" height="20" rx="4" fill="#0f172a" stroke={color} strokeWidth="1" opacity="0.9" />
              <text x="0" y="3" textAnchor="middle" fill="#f8fafc" fontSize="10" fontWeight="bold" fontFamily="monospace">{w > 0 ? '+' : ''}{w.toFixed(2)}</text>
            </g>
          )}
        </g>
      );
    });
  };

  return (
    <div className="bg-slate-800/40 backdrop-blur-md border border-slate-700/50 p-6 md:p-8 rounded-3xl shadow-xl w-full relative overflow-hidden mb-12">
      <div className="absolute top-0 right-0 w-64 h-64 bg-cyan-500/5 rounded-full blur-[80px] pointer-events-none"></div>

      <div className="mb-6">
        <h4 className="text-2xl font-bold text-slate-200">{title}</h4>
        <p className="text-slate-400 text-sm mt-1">{desc}</p>
      </div>

      <div className="w-full relative z-10">
        <div className="w-full">
          <svg viewBox="0 0 1350 600" className="w-full h-auto">
            {drawEdges(200, 500, 'in', 0, 3, 4)}
            {drawEdges(500, 800, 'h1', 12, 4, 4)}
            {drawEdges(800, 1100, 'h2', 28, 4, 4)}

            {drawRecurrentLoops(500, 'h1', 0)}
            {drawRecurrentLoops(800, 'h2', 4)}

            {INPUT_LABELS.map((label, i) => (
              <g key={`in-${i}`} onMouseEnter={() => setHoveredNode({ layer: 'in', index: i })} onMouseLeave={() => setHoveredNode(null)} className="cursor-pointer">
                <text x={165} y={155 + i * 145} textAnchor="end" alignmentBaseline="middle" fill={hoveredNode?.layer === 'in' && hoveredNode.index === i ? '#10b981' : '#cbd5e1'} fontSize={14} fontWeight="bold" className="transition-colors">{label}</text>
                <circle cx={200} cy={155 + i * 145} r={22} fill="#10b981" stroke={hoveredNode?.layer === 'in' && hoveredNode.index === i ? '#ffffff' : '#0f172a'} strokeWidth={2} className="transition-all drop-shadow-md" />
              </g>
            ))}

            {Array.from({ length: 4 }).map((_, j) => (
              <g key={`h1-${j}`} onMouseEnter={() => setHoveredNode({ layer: 'h1', index: j })} onMouseLeave={() => setHoveredNode(null)} className="cursor-pointer">
                {isLearning && activeNodes.h1.includes(j) && <circle cx={500} cy={110 + j * 125} r={32} fill="#94a3b8" opacity={0.4} className="animate-pulse" />}
                <circle cx={500} cy={110 + j * 125} r={22} fill="#64748b" stroke={hoveredNode?.layer === 'h1' && hoveredNode.index === j ? '#ffffff' : '#0f172a'} strokeWidth={2} className="transition-all drop-shadow-md" />
                <text x={500} y={110 + j * 125 + 38} textAnchor="middle" fill="#94a3b8" fontSize={12} fontFamily="monospace" fontWeight="bold">b: {biases[j].toFixed(2)}</text>
              </g>
            ))}

            {Array.from({ length: 4 }).map((_, j) => (
              <g key={`h2-${j}`} onMouseEnter={() => setHoveredNode({ layer: 'h2', index: j })} onMouseLeave={() => setHoveredNode(null)} className="cursor-pointer">
                {isLearning && activeNodes.h2.includes(j) && <circle cx={800} cy={110 + j * 125} r={32} fill="#94a3b8" opacity={0.4} className="animate-pulse" />}
                <circle cx={800} cy={110 + j * 125} r={22} fill="#64748b" stroke={hoveredNode?.layer === 'h2' && hoveredNode.index === j ? '#ffffff' : '#0f172a'} strokeWidth={2} className="transition-all drop-shadow-md" />
                <text x={800} y={110 + j * 125 + 38} textAnchor="middle" fill="#94a3b8" fontSize={12} fontFamily="monospace" fontWeight="bold">b: {biases[4 + j].toFixed(2)}</text>
              </g>
            ))}

            {OUTPUT_LABELS.map((label, k) => (
              <g key={`out-${k}`} onMouseEnter={() => setHoveredNode({ layer: 'out', index: k })} onMouseLeave={() => setHoveredNode(null)} className="cursor-pointer">
                {isLearning && activeNodes.out.includes(k) && <circle cx={1100} cy={110 + k * 125} r={32} fill="#10b981" opacity={0.4} className="animate-pulse" />}
                <circle cx={1100} cy={110 + k * 125} r={22} fill="#10b981" stroke={hoveredNode?.layer === 'out' && hoveredNode.index === k ? '#ffffff' : '#0f172a'} strokeWidth={2} className="transition-all drop-shadow-md" />
                <text x={1135} y={110 + k * 125} textAnchor="start" alignmentBaseline="middle" fill={hoveredNode?.layer === 'out' && hoveredNode.index === k ? '#10b981' : '#f8fafc'} fontSize={15} fontWeight="bold" className="transition-colors">{label}</text>
                <text x={1100} y={110 + k * 125 + 38} textAnchor="middle" fill="#94a3b8" fontSize={12} fontFamily="monospace" fontWeight="bold">b: {biases[8 + k].toFixed(2)}</text>
              </g>
            ))}

            <text x={200} y={30} textAnchor="middle" fill="#cbd5e1" fontSize={14} fontWeight="bold" className="uppercase tracking-widest">Input State</text>
            <text x={500} y={30} textAnchor="middle" fill="#cbd5e1" fontSize={14} fontWeight="bold" className="uppercase tracking-widest">Hidden Layer 1 (Recurrent)</text>
            <text x={800} y={30} textAnchor="middle" fill="#cbd5e1" fontSize={14} fontWeight="bold" className="uppercase tracking-widest">Hidden Layer 2 (Recurrent)</text>
            <text x={1100} y={30} textAnchor="middle" fill="#cbd5e1" fontSize={14} fontWeight="bold" className="uppercase tracking-widest">Action Logits</text>
          </svg>
        </div>
      </div>
    </div>
  );
};

// --- Master Dashboard Controller ---
const MultiAgentDashboard = () => {
  const [isLearning, setIsLearning] = useState(false);

  return (
    <div className="w-full flex flex-col items-center">
      <div className="flex flex-col items-center mb-12 relative z-10 w-full max-w-4xl bg-slate-800/60 backdrop-blur-xl border border-slate-700/50 p-8 rounded-3xl shadow-2xl">
        <h3 className="text-3xl md:text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-400 tracking-wide mb-3 text-center">
          Multi-Agent Simulation Core
        </h3>
        <p className="text-slate-400 text-sm mb-8 uppercase tracking-widest font-semibold text-center max-w-xl">
          Simultaneously comparing localized reinforcement learning policies against long-term memory models. Hover over nodes to inspect deep parameters.
        </p>

        <button
          onClick={() => setIsLearning(!isLearning)}
          className={`px-10 py-4 rounded-xl font-black tracking-widest text-sm uppercase transition-all duration-300 ${isLearning
              ? 'bg-rose-500/10 text-rose-400 border border-rose-500/50 hover:bg-rose-500/20 shadow-[0_0_30px_rgba(244,63,94,0.2)]'
              : 'bg-emerald-500 text-slate-950 hover:bg-emerald-400 hover:-translate-y-1 shadow-[0_0_30px_rgba(16,185,129,0.4)]'
            }`}
        >
          {isLearning ? 'Halt Global Training Feed' : 'Initiate Simultaneous Live Training'}
        </button>
      </div>

      <FeedforwardNetwork
        title="Model A: Baseline QMIX Policy"
        desc="Standard feedforward architecture optimizing for immediate queue reduction."
        isLearning={isLearning}
        seed={1}
      />
      <FeedforwardNetwork
        title="Model B: Aggressive Throughput Policy"
        desc="Alternative weight initialization demonstrating a learned preference for extending existing green phases."
        isLearning={isLearning}
        seed={42}
      />
      <RecurrentNetwork
        title="Model C: 2-Layer LSTM Predictive Model"
        desc="Deep recurrent architecture. The dotted blue feedback loops allow the network to 'remember' past signal states and predict incoming traffic waves."
        isLearning={isLearning}
      />
    </div>
  );
};

// --- Main Page Export ---
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
        <MultiAgentDashboard />
      </ScrollReveal>

      <ScrollReveal>
        <section className="flex flex-col items-center text-center max-w-4xl mx-auto mt-20">
          <h3 className="text-4xl font-bold text-slate-200 mb-6">Our Mission</h3>
          <p className="text-slate-400 text-xl leading-relaxed">
            To develop an AI-driven traffic management system that leverages real-time data from cameras and IoT sensors to optimize traffic signals, reduce congestion, and improve the daily commuting experience in urban areas.
          </p>
        </section>
      </ScrollReveal>

      <ScrollReveal>
        <section className="flex flex-col items-center text-center max-w-4xl mx-auto pb-20">
          <h3 className="text-4xl font-bold text-slate-200 mb-6">Our Vision</h3>
          <p className="text-slate-400 text-xl leading-relaxed">
            To build smarter, safer, and more sustainable cities by transforming traditional traffic control into an intelligent, adaptive ecosystem that minimizes travel time, lowers emissions, and enhances quality of life for all citizens.
          </p>
        </section>
      </ScrollReveal>
    </div>
  );
}