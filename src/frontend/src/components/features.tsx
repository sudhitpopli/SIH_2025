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

// --- Reusable Accordion Component for Future Enhancements ---
const Accordion = ({ title, content, isOpen, onClick }: { title: string, content: string, isOpen: boolean, onClick: () => void }) => (
  <div className="border border-slate-700/50 bg-slate-800/30 rounded-2xl overflow-hidden backdrop-blur-sm transition-all duration-300">
    <button 
      onClick={onClick} 
      className="w-full px-6 py-4 flex justify-between items-center text-left hover:bg-slate-700/30 transition-colors"
    >
      <span className="text-lg font-bold text-slate-200">{title}</span>
      <svg className={`w-5 h-5 text-emerald-400 transform transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
    </button>
    <div className={`transition-all duration-500 ease-in-out ${isOpen ? 'max-h-40 opacity-100' : 'max-h-0 opacity-0'}`}>
      <div className="px-6 pb-4 text-slate-400 leading-relaxed">
        {content}
      </div>
    </div>
  </div>
);

export default function Features() {
  const [openAccordion, setOpenAccordion] = useState<number | null>(0);

  const coreFeatures = [
    { title: "Dynamic Lane Priority", desc: "Detects lane density and grants green time to the busiest lane; no confusing UI, just smarter cycles." },
    { title: "Emergency Detection", desc: "Recognizes uploaded vehicle profiles and clears a path instantly for ambulances, fire trucks and police." },
    { title: "Green Corridor", desc: "Manual override for continuous green lanes; simple toggle for control room operators." },
    { title: "Obstacle Detection", desc: "Real-time detection of pedestrians/animals entering other lanes; reduces collision risk by pausing flow." },
    { title: "Emergency Handling", desc: "Automatically detect and prioritize emergency situations for rapid response and safety." },
    { title: "Map View", desc: "Real-time interactive maps showing traffic flow and incidents for better monitoring." },
    { title: "Manual Control", desc: "Allow traffic police to manually override signals and manage traffic flow as needed." },
    { title: "Reports", desc: "Generate detailed logs and traffic reports to analyze patterns and improve management." }
  ];

  const futureEnhancements = [
    { title: "AI-based traffic prediction", desc: "Use historical and live data to forecast congestion and preemptively adjust signals to avoid jams." },
    { title: "CCTV & city camera integration", desc: "Seamlessly tap into existing city infrastructure to expand vision network without new hardware." },
    { title: "IoT & connected-vehicle features", desc: "Communicate directly with smart cars to provide speed advisories and upcoming signal states." },
    { title: "Officer mobile console", desc: "Secure mobile app allowing verified personnel to trigger green corridors directly from their cruisers." }
  ];

  return (
    <div className="w-full max-w-7xl mx-auto py-8 flex flex-col gap-24">
      
      {/* --- 1. Hero Section --- */}
      <ScrollReveal>
        <section className="text-center flex flex-col items-center">
          <h2 className="text-5xl md:text-6xl font-black text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-400 tracking-tight mb-8">
            Smart Traffic Features
          </h2>
          
          {/* THE FIX: Both blocks now share the same balanced, glowing design */}
          <div className="flex flex-col md:flex-row gap-6 md:gap-8 w-full max-w-4xl">
            <div className="bg-emerald-900/10 border border-emerald-500/20 p-8 rounded-3xl flex-1 backdrop-blur-md shadow-[0_0_20px_rgba(16,185,129,0.05)] text-left flex flex-col justify-center">
              <h3 className="text-emerald-400 font-bold text-2xl mb-3">AI-Assisted Control</h3>
              <p className="text-slate-300 text-lg leading-relaxed">Reduce congestion, prioritize emergencies, and proactively prevent accidents.</p>
            </div>
            
            <div className="bg-emerald-900/10 border border-emerald-500/20 p-8 rounded-3xl flex-1 backdrop-blur-md shadow-[0_0_20px_rgba(16,185,129,0.05)] text-left flex flex-col justify-center">
              <h3 className="text-emerald-400 font-bold text-2xl mb-3">Real world focused</h3>
              <p className="text-slate-300 text-lg leading-relaxed">Simple UI, incredibly easy for control room officers to operate.</p>
            </div>
          </div>
        </section>
      </ScrollReveal>

      {/* --- 2. Core Features Grid --- */}
      <section>
        <ScrollReveal>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {coreFeatures.map((feature, index) => (
              /* THE FIX: Removed the awkward line, added a smooth title color change on hover */
              <div key={index} className="bg-slate-800/40 backdrop-blur-md border border-slate-700/50 p-8 rounded-3xl hover:-translate-y-1 hover:border-emerald-500/40 hover:shadow-[0_10px_30px_rgba(16,185,129,0.15)] transition-all duration-300 group flex flex-col justify-start">
                <h4 className="text-xl font-bold text-slate-200 mb-4 group-hover:text-emerald-400 transition-colors duration-300">
                  {feature.title}
                </h4>
                <p className="text-slate-400 text-sm leading-relaxed">
                  {feature.desc}
                </p>
              </div>
            ))}
          </div>
        </ScrollReveal>
      </section>

      {/* --- 3. How It Works Pipeline --- */}
      <section>
        <ScrollReveal>
          <h3 className="text-3xl font-bold text-slate-200 mb-10 text-center">How it works?</h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 relative">
            <div className="hidden md:block absolute top-1/2 left-0 w-full h-0.5 bg-slate-700/50 -translate-y-1/2 z-0"></div>
            
            {[
              { step: 1, title: "Monitor", desc: "Cameras / sensors collect per-lane vehicle counts and detect movement patterns in real-time." },
              { step: 2, title: "Detect", desc: "AI model recognizes emergency vehicles, pedestrians, animals, and other anomalies." },
              { step: 3, title: "Decide", desc: "System determines lane priority and decides whether to maintain cycle, switch, or override." },
              { step: 4, title: "Act & Log", desc: "Signals update instantly; events are logged for audit and later analysis." }
            ].map((item, i) => (
              <div key={i} className="relative z-10 bg-[#0B1121] md:bg-transparent border border-slate-700/50 md:border-none p-6 md:p-0 rounded-2xl flex flex-col items-center text-center">
                <div className="w-14 h-14 bg-gradient-to-br from-emerald-400 to-cyan-500 rounded-2xl flex items-center justify-center text-slate-950 font-black text-2xl mb-6 shadow-lg shadow-emerald-500/20">
                  {item.step}
                </div>
                <h4 className="text-xl font-bold text-slate-200 mb-3">{item.title}</h4>
                <p className="text-slate-400 text-sm">{item.desc}</p>
              </div>
            ))}
          </div>
        </ScrollReveal>
      </section>

      {/* --- 4. Why Choose Us & Future --- */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-start pb-20">
        <ScrollReveal>
          <div>
            <h3 className="text-3xl font-bold text-slate-200 mb-8">Why choose Smart Traffic?</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {[
                { title: "Saves Time", desc: "Less idle time at signals — more consistent throughput for busy intersections." },
                { title: "Prioritizes Emergencies", desc: "Faster response for ambulances and fire units — critical minutes saved." },
                { title: "Safer Roads", desc: "Stops traffic when obstructions appear, reducing collision risk and injuries." },
                { title: "Data Driven", desc: "Logs and reports give police insights to improve signal timing and city planning." }
              ].map((item, i) => (
                <div key={i} className="bg-slate-800/40 border border-slate-700/50 p-6 rounded-3xl hover:bg-slate-800/60 transition-colors">
                  <h4 className="text-emerald-400 font-bold mb-3">{item.title}</h4>
                  <p className="text-slate-400 text-sm leading-relaxed">{item.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </ScrollReveal>

        <ScrollReveal delay={200}>
          <div className="flex flex-col gap-4">
            <h3 className="text-3xl font-bold text-slate-200 mb-4">Future Enhancements</h3>
            {futureEnhancements.map((enhancement, i) => (
              <Accordion 
                key={i} 
                title={enhancement.title} 
                content={enhancement.desc} 
                isOpen={openAccordion === i} 
                onClick={() => setOpenAccordion(openAccordion === i ? null : i)} 
              />
            ))}
          </div>
        </ScrollReveal>
      </section>

    </div>
  );
}