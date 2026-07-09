import React from 'react';
import { Search } from 'lucide-react';
import { cn } from '@/lib/utils';

export const AnalysisProgress = ({ progress, stage }) => {
  return (
    <div className="w-full space-y-4 animate-fade-in py-2">
      
      {/* Top Status Area */}
      <div className="flex items-center justify-between border-b border-[#d4af37]/20 pb-2">
        <div className="flex items-center gap-3">
          {/* Magnifying Glass acting as the scanning detective */}
          <div className="relative">
            <Search className="w-5 h-5 text-[#d4af37] animate-pulse" />
            <div className="absolute inset-0 bg-[#d4af37]/20 blur-sm rounded-full animate-ping" />
          </div>
          <span className="text-xs font-mono tracking-widest uppercase text-[#d4af37]">
            {stage}...
          </span>
        </div>
        <span className="text-xs font-mono tracking-widest text-[#d4af37] font-bold">
          {Math.round(progress)}%
        </span>
      </div>
      
      {/* Forensic Slashed Progress Bar */}
      <div className="relative h-4 w-full bg-black/60 rounded-sm border border-[#d4af37]/30 overflow-hidden shadow-inner">
        {/* The Fill */}
        <div 
          className="absolute top-0 left-0 h-full bg-[#d4af37] transition-all duration-300 ease-out flex justify-end"
          style={{ width: `${progress}%` }}
        >
          {/* Diagonal Slashes Overlay (Warning Tape / Pencil Hash Vibe) */}
          <div 
            className="absolute inset-0 opacity-40"
            style={{
              backgroundImage: 'repeating-linear-gradient(-45deg, transparent, transparent 4px, rgba(0,0,0,0.8) 4px, rgba(0,0,0,0.8) 8px)'
            }}
          />
          
          {/* Glowing Scanner Beam at the leading edge */}
          <div className="relative w-1.5 h-full bg-white shadow-[0_0_8px_2px_#fff,0_0_12px_4px_#d4af37] z-10" />
        </div>
      </div>
      
      {/* Bottom Forensic Waypoints */}
      <div className="flex justify-between text-[10px] text-[#8c7b5a] font-mono tracking-widest uppercase opacity-70">
        <span className={cn(progress >= 0 ? "text-[#d4af37] opacity-100" : "")}>[INIT]</span>
        <span className={cn(progress >= 25 ? "text-[#d4af37] opacity-100" : "")}>[EXTRACT]</span>
        <span className={cn(progress >= 75 ? "text-[#d4af37] opacity-100" : "")}>[ANALYZE]</span>
        <span className={cn(progress >= 100 ? "text-[#d4af37] opacity-100" : "")}>[DONE]</span>
      </div>
      
    </div>
  );
};
