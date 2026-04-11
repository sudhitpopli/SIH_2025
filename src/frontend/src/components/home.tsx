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
    }, { threshold: 0.1 });

    if (domRef.current) observer.observe(domRef.current);
    return () => observer.disconnect();
  }, []);

  return (
    <div
      ref={domRef}
      className={`transition-all duration-1000 ease-out ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-16'}`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {children}
    </div>
  );
};

export default function Home({ onNavigate }: { onNavigate: (tab: string) => void }) {
  return (
    <div className="w-full max-w-7xl mx-auto py-8 flex flex-col gap-32">
      
      {/* --- 1. The Big Welcome (Hero Section) --- */}
      <section className="relative w-full min-h-[60vh] flex flex-col items-center justify-center text-center px-4">
        
        {/* Abstract Glowing Background Rings */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-emerald-500/10 rounded-full blur-[100px] -z-10 pointer-events-none"></div>
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[400px] h-[400px] bg-cyan-500/10 rounded-full blur-[80px] -z-10 pointer-events-none"></div>

        <ScrollReveal>
          <div className="inline-block mb-6 px-4 py-1.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 text-emerald-400 font-semibold tracking-widest text-sm uppercase backdrop-blur-md">
            System Online • Version 1.0
          </div>
          <h1 className="text-6xl md:text-8xl font-black text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-400 tracking-tight mb-8 leading-tight">
            The Future of <br className="hidden md:block" /> Urban Mobility
          </h1>
          <p className="text-slate-300 text-xl md:text-2xl leading-relaxed max-w-3xl mx-auto mb-12">
            Transforming rigid, outdated stoplights into a living, learning ecosystem. QMIX uses reinforcement learning to cure city congestion.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-6 justify-center">
            <button 
              onClick={() => onNavigate('Dashboard')}
              className="px-8 py-4 rounded-2xl font-bold tracking-wide text-lg bg-emerald-500 text-slate-950 hover:bg-emerald-400 hover:-translate-y-1 shadow-[0_0_20px_rgba(16,185,129,0.3)] transition-all duration-300"
            >
              Enter Command Center
            </button>
            <button 
              onClick={() => onNavigate('Features')}
              className="px-8 py-4 rounded-2xl font-bold tracking-wide text-lg bg-slate-800/50 text-slate-200 border border-slate-700/50 hover:bg-slate-700/50 hover:-translate-y-1 transition-all duration-300 backdrop-blur-md"
            >
              Explore Capabilities
            </button>
          </div>
        </ScrollReveal>
      </section>

      {/* --- 2. The Problem (Did you know?) --- */}
      <section className="relative">
        <ScrollReveal>
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-black text-slate-200 tracking-tight mb-4">
              The Cost of Inaction
            </h2>
            <p className="text-slate-400 text-lg max-w-2xl mx-auto">
              Current static traffic systems are failing our growing cities. The consequences are measurable in time, money, and lives.
            </p>
          </div>
        </ScrollReveal>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          
          {/* Stat 1: Time */}
          <ScrollReveal delay={0}>
            <div className="bg-rose-900/10 backdrop-blur-md border border-rose-500/20 p-8 rounded-3xl hover:-translate-y-1 hover:shadow-[0_10px_30px_rgba(244,63,94,0.1)] transition-all duration-300 h-full flex flex-col justify-center">
              <h3 className="text-5xl font-black text-rose-400 mb-4">99 <span className="text-2xl text-slate-400 font-bold">hrs</span></h3>
              <h4 className="text-xl font-bold text-slate-200 mb-3">Lost Annually</h4>
              <p className="text-slate-400 leading-relaxed">
                The average urban commuter spends nearly 100 hours a year idling at red lights. Static timers cannot adapt to rush hour surges.
              </p>
            </div>
          </ScrollReveal>

          {/* Stat 2: Environment */}
          <ScrollReveal delay={150}>
            <div className="bg-rose-900/10 backdrop-blur-md border border-rose-500/20 p-8 rounded-3xl hover:-translate-y-1 hover:shadow-[0_10px_30px_rgba(244,63,94,0.1)] transition-all duration-300 h-full flex flex-col justify-center">
              <h3 className="text-5xl font-black text-rose-400 mb-4">3B <span className="text-2xl text-slate-400 font-bold">gal</span></h3>
              <h4 className="text-xl font-bold text-slate-200 mb-3">Wasted Fuel</h4>
              <p className="text-slate-400 leading-relaxed">
                Billions of gallons of fuel are burned solely due to traffic congestion, contributing massively to urban carbon footprints and poor air quality.
              </p>
            </div>
          </ScrollReveal>

          {/* Stat 3: Safety */}
          <ScrollReveal delay={300}>
            <div className="bg-rose-900/10 backdrop-blur-md border border-rose-500/20 p-8 rounded-3xl hover:-translate-y-1 hover:shadow-[0_10px_30px_rgba(244,63,94,0.1)] transition-all duration-300 h-full flex flex-col justify-center">
              <h3 className="text-5xl font-black text-rose-400 mb-4">4 <span className="text-2xl text-slate-400 font-bold">min</span></h3>
              <h4 className="text-xl font-bold text-slate-200 mb-3">Response Delay</h4>
              <p className="text-slate-400 leading-relaxed">
                Current rigid signal cycles delay emergency responders by an average of 3-4 minutes, a margin that often determines life or death outcomes.
              </p>
            </div>
          </ScrollReveal>

        </div>
      </section>

    </div>
  );
}