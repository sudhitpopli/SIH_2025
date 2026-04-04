import { useEffect, useRef } from 'react';
import type { Vehicle } from '../hooks/useTrafficSocket';

interface SimulationMapProps {
  vehicles: Vehicle[];
  isActive: boolean; // true for V2, false for Native to change colors
  title: string;
  mapRoads?: number[][][]; // Polygons from SUMO
}

export default function SimulationMap({ vehicles, isActive, title, mapRoads }: SimulationMapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // We want to zoom in on a small section of the map where cars are actually concentrated.
  // We strictly EXPAND the bounding box. Once the simulation runs the outer edges of the map,
  // the camera will perfectly freeze in place, allowing us to draw trails without blurring.
  const boundsRef = useRef({
    minX: 100000,
    maxX: -100000,
    minY: 100000,
    maxY: -100000
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;


    if (vehicles.length > 0) {
      let cMinX = Infinity, cMaxX = -Infinity, cMinY = Infinity, cMaxY = -Infinity;
      vehicles.forEach(v => {
        if (v.x < cMinX) cMinX = v.x;
        if (v.x > cMaxX) cMaxX = v.x;
        if (v.y < cMinY) cMinY = v.y;
        if (v.y > cMaxY) cMaxY = v.y;
      });

      // Expand camera statically
      const padX = 200;
      const padY = 200;
      if (cMinX - padX < boundsRef.current.minX) boundsRef.current.minX = cMinX - padX;
      if (cMaxX + padX > boundsRef.current.maxX) boundsRef.current.maxX = cMaxX + padX;
      if (cMinY - padY < boundsRef.current.minY) boundsRef.current.minY = cMinY - padY;
      if (cMaxY + padY > boundsRef.current.maxY) boundsRef.current.maxY = cMaxY + padY;
    }

    let { minX, maxX, minY, maxY } = boundsRef.current;

    // Default bounds override ONLY for the drawing sequence so the map renders massively zoomed out on boot.
    // We strictly DO NOT mutate `boundsRef.current` so the auto-snapping logic works perfectly once cars spawn!
    if (minX >= maxX) {
      minX = 0;
      maxX = 6520;
      minY = 0;
      maxY = 11330;
    }

    const mapWidth = maxX - minX;
    const mapHeight = maxY - minY;

    const width = canvas.width;
    const height = canvas.height;

    // MAGICAL FADING WIPE
    ctx.fillStyle = 'rgba(15, 23, 42, 0.12)'; 
    ctx.fillRect(0, 0, width, height);

    // DRAW THE ACTUAL SUMO ROAD MAP
    if (mapRoads && mapRoads.length > 0) {
      ctx.strokeStyle = 'rgba(30, 41, 59, 1.0)'; 
      ctx.lineWidth = 1;
      
      mapRoads.forEach(road => {
        if (road.length < 2) return;
        ctx.beginPath();
        const startNx = ((road[0][0] - minX) / mapWidth) * width;
        const startNy = height - (((road[0][1] - minY) / mapHeight) * height);
        ctx.moveTo(startNx, startNy);
        
        for (let i = 1; i < road.length; i++) {
          const nx = ((road[i][0] - minX) / mapWidth) * width;
          const ny = height - (((road[i][1] - minY) / mapHeight) * height);
          ctx.lineTo(nx, ny);
        }
        ctx.stroke();
      });
    }

    // Draw vehicles
    vehicles.forEach(v => {
      const nx = ((v.x - minX) / mapWidth) * width;
      const ny = height - (((v.y - minY) / mapHeight) * height);

      const rad = (v.phi - 90) * (Math.PI / 180);

      ctx.save();
      ctx.translate(nx, ny);
      ctx.rotate(rad);

      ctx.beginPath();
      ctx.moveTo(6, 0);
      ctx.lineTo(-4, -3);
      ctx.lineTo(-4, 3);
      ctx.closePath();

      if (isActive) {
        ctx.fillStyle = '#10b981'; 
        ctx.shadowColor = '#34d399';
        ctx.shadowBlur = 8;
      } else {
        ctx.fillStyle = '#94a3b8'; 
      }
      ctx.fill();
      ctx.restore();
    });

  }, [vehicles, isActive, mapRoads]);

  return (
    <div className="flex flex-col items-center w-full bg-slate-800/40 p-4 rounded-2xl border border-slate-700/50 shadow-xl">
      <h3 className="text-lg font-bold text-slate-200 mb-3 tracking-wider">
        {title}
      </h3>
      <canvas
        ref={canvasRef}
        width={500}
        height={500}
        className="w-full max-w-full aspect-square rounded-xl bg-slate-900 border border-slate-700"
      />
      <div className="mt-4 flex w-full justify-between px-2 text-sm text-slate-400">
        <span>Active Vehicles: <span className="font-mono text-slate-200">{vehicles.length}</span></span>
        {isActive && <span className="text-emerald-400 animate-pulse text-xs">● AI Controller Online</span>}
        {!isActive && <span className="text-slate-500 text-xs text-right">● Native Logic Online</span>}
      </div>
    </div>
  );
}
