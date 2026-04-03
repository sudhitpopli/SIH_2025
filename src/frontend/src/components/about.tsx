import { useEffect, useRef, useState } from 'react';

// --- Reusable Scroll Reveal Component ---
const ScrollReveal = ({ children, delay = 0 }: { children: React.ReactNode, delay?: number }) => {
  const [isVisible, setIsVisible] = useState(false);
  const domRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Set up the tripwire
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

export default function About() {
  return (
    <div className="w-full max-w-7xl mx-auto py-12 flex flex-col gap-32">
      
      {/* --- Section 1: Hero Introduction --- */}
      <ScrollReveal>
        <section className="text-center flex flex-col items-center">
          <h2 className="text-5xl md:text-6xl font-black text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-400 tracking-tight mb-10">
            About Our Project
          </h2>
          <p className="text-slate-300 text-xl leading-relaxed max-w-5xl">
            We are pioneering the future of urban mobility with an AI-driven traffic management system designed to optimize signal timings and reduce congestion in cities. Leveraging real-time data from cameras and IoT sensors, our platform uses cutting-edge computer vision and reinforcement learning technologies to predict bottlenecks and adapt traffic flow dynamically. Our mission is to shorten commute times, enhance road safety, and provide traffic authorities with powerful tools to monitor and control signals effectively. Committed to innovation and sustainability, we strive to create smarter, smoother, and safer roads for everyone.
          </p>
        </section>
      </ScrollReveal>

      {/* --- Section 2: Mission --- */}
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

      {/* --- Section 3: Vision --- */}
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