

interface NavbarProps {
  tabs: string[];
  activeTab: string;
  onTabChange: (tab: string) => void;
}

export default function Navbar({ tabs, activeTab, onTabChange }: NavbarProps) {
  return (
    <div className="flex flex-wrap justify-center items-center bg-slate-800/40 p-1.5 rounded-full border border-slate-700/50 mb-10 shadow-lg backdrop-blur-md">
      {tabs.map((tab) => {
        const isActive = activeTab === tab;
        return (
          <button
            key={tab}
            onClick={() => onTabChange(tab)}
            className={`relative px-6 py-2.5 mx-0.5 rounded-full text-sm font-bold tracking-wide transition-colors duration-300 z-10 overflow-hidden ${
              isActive 
                ? 'text-slate-950' 
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/30'
            }`}
          >
            <span className="relative z-10">{tab}</span>
            
            {/* The glowing active highlight pill */}
            <div 
              className={`absolute inset-0 bg-emerald-500 rounded-full -z-10 transition-all duration-300 ease-out ${
                isActive ? 'opacity-100 scale-100 shadow-[0_0_15px_rgba(16,185,129,0.4)]' : 'opacity-0 scale-75'
              }`}
            />
          </button>
        );
      })}
    </div>
  );
}