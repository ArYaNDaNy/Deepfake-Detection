import React from 'react';
import { Eye, Scan, Zap } from 'lucide-react';

export const HeroSection = () => {
  return (
    <section className="relative py-16 md:py-24 overflow-hidden w-full mt-10">
      
      <div className="relative z-10 text-center space-y-10 max-w-4xl mx-auto px-4 flex flex-col items-center">
        
        {/* Top Badge */}
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-sm border border-[#d4af37]/60 bg-black/70 backdrop-blur-md text-sm text-[#d4af37] animate-fade-in font-mono uppercase tracking-widest shadow-[0_0_15px_rgba(212,175,55,0.2)]">
          <Eye className="w-4 h-4" />
          <span>Open Your Eyes</span>
        </div>
        
        {/* INVISIBLE SPACER: This leaves a massive empty gap in the code so the "BELIEVE" text from your actual background image is perfectly visible */}
        <div className="h-32 md:h-48 w-full"></div>

        {/* Subtext Box */}
        <div className="bg-black/60 p-6 rounded-lg backdrop-blur-xl border border-[#d4af37]/40 shadow-2xl max-w-2xl mx-auto w-full">
          <p 
            className="text-lg md:text-xl text-[#c3b699] animate-fade-in font-mono tracking-widest leading-relaxed" 
            style={{ animationDelay: '0.2s' }}
          >
            "A constructed reality.<br/>Find the single grain of truth."
          </p>
        </div>
        
        {/* Features Row */}
        <div 
          className="flex flex-wrap justify-center gap-4 pt-4 animate-fade-in" 
          style={{ animationDelay: '0.3s' }}
        >
          <div className="flex items-center gap-2 px-4 py-2 bg-black/70 backdrop-blur-xl rounded border border-[#d4af37]/40 shadow-[0_4px_20px_rgba(0,0,0,0.8)]">
            <Scan className="w-4 h-4 text-[#d4af37]" />
            <span className="text-sm font-mono text-[#e8dfce] uppercase tracking-wide">Frame Analysis</span>
          </div>
          <div className="flex items-center gap-2 px-4 py-2 bg-black/70 backdrop-blur-xl rounded border border-[#d4af37]/40 shadow-[0_4px_20px_rgba(0,0,0,0.8)]">
            <Zap className="w-4 h-4 text-[#d4af37]" />
            <span className="text-sm font-mono text-[#e8dfce] uppercase tracking-wide">Instant Detection</span>
          </div>
        </div>
      </div>
    </section>
  );
};