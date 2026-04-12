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

export default function Contact() {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSent, setIsSent] = useState(false);

  // Mock submission handler for the demo
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setTimeout(() => {
      setIsSubmitting(false);
      setIsSent(true);
      setTimeout(() => setIsSent(false), 3000); // Reset after 3 seconds
    }, 1500);
  };

  return (
    <div className="w-full max-w-7xl mx-auto py-8 flex flex-col gap-16 pb-24">

      {/* --- 1. Hero Section --- */}
      <ScrollReveal>
        <section className="text-center flex flex-col items-center">
          <h2 className="text-5xl md:text-6xl font-black text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-400 tracking-tight mb-6">
            Contact Us
          </h2>
          <p className="text-slate-300 text-xl leading-relaxed max-w-3xl mx-auto">
            Feel free to reach out to our team regarding our QMIX Model
          </p>
        </section>
      </ScrollReveal>

      {/* --- 2. Contact Grid --- */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-12 items-start">

        {/* Left Column: Contact Information (Takes up 2/5 of the grid) */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          <ScrollReveal delay={100}>
            <div className="bg-slate-800/40 backdrop-blur-md border border-slate-700/50 p-8 rounded-3xl hover:border-emerald-500/30 transition-colors group">
              <div className="w-14 h-14 bg-emerald-500/10 rounded-2xl flex items-center justify-center mb-6 border border-emerald-500/20 group-hover:scale-110 transition-transform">
                <svg className="w-7 h-7 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" /></svg>
              </div>
              <h4 className="text-xl font-bold text-slate-200 mb-2">Our Email</h4>
              <p className="text-emerald-400 font-mono text-sm mb-1">sudhitpopli@gmail.com</p>
            </div>
          </ScrollReveal>

          <ScrollReveal delay={200}>
            <div className="bg-slate-800/40 backdrop-blur-md border border-slate-700/50 p-8 rounded-3xl hover:border-cyan-500/30 transition-colors group">
              <div className="w-14 h-14 bg-cyan-500/10 rounded-2xl flex items-center justify-center mb-6 border border-cyan-500/20 group-hover:scale-110 transition-transform">
                <svg className="w-7 h-7 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" /></svg>
              </div>
              <h4 className="text-xl font-bold text-slate-200 mb-2">Contact Number</h4>
              <p className="text-cyan-400 font-mono text-sm mb-1">+91 9876543210</p>
              <p className="text-slate-500 text-sm">Mon-Fri, 09:00 - 18:00 IST</p>
            </div>
          </ScrollReveal>


        </div>

        {/* Right Column: The Form (Takes up 3/5 of the grid) */}
        <div className="lg:col-span-3">
          <ScrollReveal delay={150}>
            <div className="bg-slate-800/40 backdrop-blur-md border border-slate-700/50 p-8 md:p-12 rounded-3xl shadow-xl relative overflow-hidden">

              {/* Decorative background glow */}
              <div className="absolute top-0 right-0 -translate-y-1/2 translate-x-1/3 w-64 h-64 bg-emerald-500/10 rounded-full blur-[60px] pointer-events-none"></div>

              <h3 className="text-2xl font-bold text-slate-200 mb-8">Send a Report</h3>

              <form onSubmit={handleSubmit} className="flex flex-col gap-6 relative z-10">

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="flex flex-col gap-2">
                    <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Operator Name</label>
                    <input
                      type="text"
                      required
                      placeholder="Jane Doe"
                      className="bg-slate-900/50 border border-slate-700/50 text-slate-200 rounded-xl px-4 py-3 focus:outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/50 transition-all placeholder:text-slate-600"
                    />
                  </div>
                  <div className="flex flex-col gap-2">
                    <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Agency / Department</label>
                    <input
                      type="text"
                      required
                      placeholder="City Transit Authority"
                      className="bg-slate-900/50 border border-slate-700/50 text-slate-200 rounded-xl px-4 py-3 focus:outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/50 transition-all placeholder:text-slate-600"
                    />
                  </div>
                </div>

                <div className="flex flex-col gap-2">
                  <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Encrypted Email Address</label>
                  <input
                    type="email"
                    required
                    placeholder="jane@citytransit.gov"
                    className="bg-slate-900/50 border border-slate-700/50 text-slate-200 rounded-xl px-4 py-3 focus:outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/50 transition-all placeholder:text-slate-600"
                  />
                </div>

                <div className="flex flex-col gap-2">
                  <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Mission Details</label>
                  <textarea
                    required
                    rows={5}
                    placeholder="Describe your city's current traffic bottlenecks..."
                    className="bg-slate-900/50 border border-slate-700/50 text-slate-200 rounded-xl px-4 py-3 focus:outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/50 transition-all resize-none placeholder:text-slate-600"
                  ></textarea>
                </div>

                <button
                  type="submit"
                  disabled={isSubmitting || isSent}
                  className={`mt-4 px-8 py-4 rounded-xl font-bold tracking-widest text-sm uppercase transition-all duration-300 ${isSent
                      ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/50 cursor-not-allowed'
                      : isSubmitting
                        ? 'bg-slate-700 text-slate-400 cursor-wait'
                        : 'bg-emerald-500 text-slate-950 hover:bg-emerald-400 hover:-translate-y-1 shadow-[0_0_20px_rgba(16,185,129,0.3)]'
                    }`}
                >
                  {isSent ? 'Report Sent ✓' : isSubmitting ? 'Reporting...' : 'Send Report'}
                </button>
              </form>
            </div>
          </ScrollReveal>
        </div>

      </div>
    </div>
  );
}