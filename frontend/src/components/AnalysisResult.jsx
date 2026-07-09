import React from 'react';
import { Paperclip, Fingerprint, Mic } from 'lucide-react';
import { cn } from '@/lib/utils';

export const AnalysisResult = ({ result }) => {
  const isFake = result.isDeepfake;

  const caseNumber = React.useMemo(() => 
    `DF-${Math.floor(Math.random() * 90000) + 10000}-26`, []
  );

  const today = new Date().toLocaleDateString('en-US', {
    year: 'numeric', month: 'short', day: '2-digit'
  });

  return (
    <div className="w-full max-w-3xl mx-auto pb-10 mt-8 relative">
      
      {/* Background Pages */}
      <div className="absolute top-0 left-0 right-0 bottom-4 bg-[#dfd5b4] -translate-x-2 translate-y-4 -rotate-1 border border-[#d2c7aa] shadow-md z-0"></div>
      <div className="absolute top-0 left-0 right-0 bottom-8 bg-[#e8debf] translate-x-2 translate-y-2 rotate-1 border border-[#d2c7aa] shadow-sm z-0"></div>

      {/* 
        Main Page 
        FIXED: Increased bottom padding from pb-48 to pb-72/pb-80 to guarantee the Polaroid NEVER touches the text above it.
      */}
      <div className="relative bg-[#f4ebd0] p-8 md:p-12 pl-16 md:pl-20 pb-72 md:pb-80 shadow-[5px_15px_40px_rgba(0,0,0,0.8)] text-slate-900 border border-[#d2c7aa] min-h-[600px] z-10 overflow-hidden">
        
        {/* Binder Spine */}
        <div className="absolute top-0 bottom-0 left-0 w-8 md:w-12 bg-[#2a2d34] border-r border-slate-900 shadow-[inset_-4px_0_15px_rgba(0,0,0,0.8)] z-30 flex flex-col justify-evenly items-center">
            <div className="w-3 h-3 md:w-4 md:h-4 rounded-full bg-[#1c1814] shadow-[inset_2px_2px_4px_rgba(0,0,0,0.9)]"></div>
            <div className="w-3 h-3 md:w-4 md:h-4 rounded-full bg-[#1c1814] shadow-[inset_2px_2px_4px_rgba(0,0,0,0.9)]"></div>
            <div className="w-3 h-3 md:w-4 md:h-4 rounded-full bg-[#1c1814] shadow-[inset_2px_2px_4px_rgba(0,0,0,0.9)]"></div>
        </div>

        {/* Paperclip */}
        <div className="absolute -top-6 left-16 md:left-24 text-slate-500 -rotate-12 drop-shadow-lg z-40 pointer-events-none">
          <Paperclip className="w-14 h-14" />
        </div>

        <div className="absolute top-0 bottom-0 left-8 md:left-12 w-10 bg-gradient-to-r from-transparent via-black/[0.04] to-transparent pointer-events-none mix-blend-multiply z-20"></div>

        {/* Confidential Stamp */}
        <div className="absolute top-8 right-8 border-2 border-red-800/40 text-red-800/40 px-2 py-1 font-mono text-sm font-bold tracking-widest uppercase rotate-2 z-10 mix-blend-multiply pointer-events-none">
          Strictly Confidential
        </div>

        {/* Header */}
        <div className="flex justify-between items-start border-b-2 border-slate-800 pb-4 mb-8 mt-4 relative z-10">
          <div className="flex flex-col">
            <span className="font-mono text-xl md:text-2xl font-bold uppercase tracking-widest text-slate-900">
              Dept. of Digital Forensics
            </span>
            <span className="font-mono text-sm tracking-widest text-slate-700">
              Media Analysis Division
            </span>
          </div>
          <div className="flex flex-col text-right font-mono text-sm mt-1">
            <span><strong>CASE NO:</strong> {caseNumber}</span>
            <span><strong>DATE:</strong> {today}</span>
          </div>
        </div>

        {/* Form Data */}
        <div className="space-y-6 mb-12 relative z-10">
          <div className="flex items-end border-b border-slate-400/70 pb-1">
            <span className="font-mono font-bold w-32 md:w-40 shrink-0 text-slate-800 text-sm md:text-base">SUBJECT FILE:</span>
            <span className="font-[cursive] text-xl md:text-2xl text-[#2a2a27] -rotate-1 ml-4 tracking-wide break-all">
              {result.fileName}
            </span>
          </div>
          <div className="flex items-end border-b border-slate-400/70 pb-1">
            <span className="font-mono font-bold w-32 md:w-40 shrink-0 text-slate-800 text-sm md:text-base">MODALITY:</span>
            <span className="font-[cursive] text-xl md:text-2xl text-[#2a2a27] -rotate-1 ml-4 tracking-wide uppercase">
              {result.fileType} ANALYSIS
            </span>
          </div>
          <div className="flex items-end border-b border-slate-400/70 pb-1">
            <span className="font-mono font-bold w-32 md:w-40 shrink-0 text-slate-800 text-sm md:text-base">THREAT LEVEL:</span>
            <span className="font-[cursive] text-xl md:text-2xl text-[#2a2a27] ml-4 tracking-wide">
              {isFake ? `${result.confidence}% MANIPULATED` : `${100 - result.confidence}% NATURAL`}
            </span>
          </div>
        </div>

        {/* Details Grid */}
        <div className="mb-8 relative z-10 w-full md:w-2/3">
          <h3 className="font-mono font-bold text-slate-900 border-b-2 border-slate-800 mb-6 inline-block tracking-wider">
            INVESTIGATOR NOTES & HEURISTICS
          </h3>
          <div className="grid grid-cols-1 gap-5">
            {result.details.map((detail, index) => (
              <div key={index} className="flex flex-col sm:flex-row sm:items-end gap-1 sm:gap-4 border-b border-slate-400/40 pb-2">
                <div className="font-mono text-sm font-semibold text-slate-700 w-48 shrink-0 flex items-center gap-2">
                  <span>►</span> {detail.label}
                </div>
                {/* FIXED: Removed `truncate` so long text like "SVM (RBF, C=50)" wraps safely */}
                <div className="font-[cursive] text-lg md:text-xl text-[#2a2a27] tracking-wide mt-1 sm:mt-0 leading-relaxed">
                  {detail.value}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="absolute bottom-8 left-16 opacity-[0.07] pointer-events-none z-10">
          <Fingerprint className="w-32 h-32 text-slate-900" />
        </div>

        {/* 
          =======================================
          THE EVIDENCE POLAROID
          FIXED: Moved down to `bottom-8` and `bottom-12`. Since we added massive padding above, 
          it will no longer touch the heuristics grid!
          =======================================
        */}
        <div className="absolute bottom-8 md:bottom-12 right-6 md:right-12 z-20 flex flex-col items-center rotate-[4deg] shadow-[4px_8px_15px_rgba(0,0,0,0.4)] bg-[#faf9f6] p-3 pb-10 border border-gray-300 w-48 h-56 md:w-56 md:h-64 transition-transform hover:scale-105 hover:rotate-0 hover:z-50 duration-300">
          
          {/* Masking tape holding the photo */}
          <div className="absolute -top-3 left-1/2 -translate-x-1/2 w-16 h-5 bg-[#e5e0d8]/80 backdrop-blur-sm -rotate-3 shadow-sm z-30 opacity-90 border border-white/20"></div>

          {/* The Media Frame */}
          <div className="w-full h-full bg-[#1c1814] flex items-center justify-center overflow-hidden shadow-[inset_0_0_8px_rgba(0,0,0,0.5)]">
            {result.fileType === 'image' && (
              <img src={result.previewUrl} alt="Evidence" className="w-full h-full object-cover" />
            )}
            
            {result.fileType === 'video' && (
              <video src={result.previewUrl} className="w-full h-full object-cover" />
            )}
            
            {result.fileType === 'audio' && (
              <div className="flex flex-col items-center gap-2 text-slate-600">
                <Mic className="w-12 h-12" />
                <span className="font-mono text-xs tracking-widest">AUDIO RECORD</span>
              </div>
            )}
          </div>
          
          {/* Handwritten tag on the Polaroid lip */}
          <div className="absolute bottom-2 font-[cursive] text-slate-700 text-sm tracking-widest -rotate-2">
            Exhibit A
          </div>

          {/* THE OVERLAPPING STAMP */}
          <div className={cn(
            "absolute -bottom-6 -left-8 flex items-center justify-center p-2 border-[5px] border-double rounded-sm rotate-[-15deg] opacity-90 shadow-sm mix-blend-multiply z-30 pointer-events-none bg-transparent backdrop-blur-[1px]",
            isFake ? "border-red-700 text-red-700" : "border-emerald-700 text-emerald-700"
          )}>
            <div className="font-mono text-xl md:text-3xl font-black uppercase tracking-tighter text-center leading-none">
              {isFake ? 'FORGERY\nDETECTED' : 'VERIFIED\nAUTHENTIC'}
            </div>
          </div>
        </div>

        {/* Paper Texture Noise */}
        <div className="absolute inset-0 pointer-events-none opacity-[0.04] z-10" 
             style={{ backgroundImage: 'url("data:image/svg+xml,%3Csvg viewBox=%220 0 200 200%22 xmlns=%22http://www.w3.org/2000/svg%22%3E%3Cfilter id=%22noiseFilter%22%3E%3CfeTurbulence type=%22fractalNoise%22 baseFrequency=%220.85%22 numOctaves=%223%22 stitchTiles=%22stitch%22/%3E%3C/filter%3E%3Crect width=%22100%25%22 height=%22100%25%22 filter=%22url(%23noiseFilter)%22/%3E%3C/svg%3E")' }}>
        </div>

      </div>
    </div>
  );
};