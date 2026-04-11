import React, { useEffect, useState, useRef } from "react";
import { Activity, Shield, Cpu, Network, Zap } from "lucide-react";

export default function LoadingScreen({ onComplete }) {
  const [progress, setProgress] = useState(0);
  const [step, setStep] = useState(0);
  
  const onCompleteRef = useRef(onComplete);
  useEffect(() => {
    onCompleteRef.current = onComplete;
  }, [onComplete]);

  const steps = [
    { label: "Establishing secure connection to core...", icon: Network },
    { label: "Loading initial context memory...", icon: Cpu },
    { label: "Warming up PPO and Dreamer models...", icon: Activity },
    { label: "Verifying risk authority...", icon: Shield },
    { label: "Connecting to live symbol lanes...", icon: Zap }
  ];

  useEffect(() => {
    const totalTime = 4000;
    const interval = 40;
    let current = 0;

    const timer = setInterval(() => {
      current += interval;
      // Use easing for smoother progress feel
      const normalizedTime = current / totalTime;
      // simple ease-out quad
      const easedProgress = Math.min(100, (1 - (1 - normalizedTime) * (1 - normalizedTime)) * 100);
      
      setProgress(easedProgress);

      if (easedProgress > 85) setStep(4);
      else if (easedProgress > 65) setStep(3);
      else if (easedProgress > 40) setStep(2);
      else if (easedProgress > 15) setStep(1);

      if (current >= totalTime) {
        clearInterval(timer);
        setTimeout(() => {
          if (onCompleteRef.current) onCompleteRef.current();
        }, 400); // small pause before transition
      }
    }, interval);

    return () => clearInterval(timer);
  }, []);

  const CurrentIcon = steps[step].icon;

  return (
    <div className="loading-screen">
      <div className="loading-content">
        <div className="logo-container">
          <div className="logo-pulse"></div>
          <div className="logo-core">
            <svg viewBox="0 0 100 100" className="brand-logo">
              {/* Outer hexagon */}
              <path d="M50 5 L88.97 27.5 L88.97 72.5 L50 95 L11.03 72.5 L11.03 27.5 Z" fill="none" stroke="var(--cyan)" strokeWidth="2.5" strokeDasharray="30 10" className="logo-path-slow" />
              {/* Inner shape */}
              <path d="M50 20 L75 35 L75 65 L50 80 L25 65 L25 35 Z" fill="none" stroke="var(--purple)" strokeWidth="1.5" strokeDasharray="4 6" className="logo-path-fast" />
              {/* Center core */}
              <circle cx="50" cy="50" r="10" fill="var(--cyan)" className="logo-dot" />
            </svg>
          </div>
        </div>
        
        <h1 className="brand-title">CAUTIOUS GIGGLE</h1>
        <div className="brand-subtitle">AUTONOMOUS TRADING PIPELINE</div>

        <div className="loading-status-area">
          <div className="loading-step-row" key={step}>
            <CurrentIcon size={16} className="loading-step-icon" />
            <span className="loading-step-text">{steps[step].label}</span>
          </div>
          
          <div className="global-progress-bar">
            <div className="global-progress-fill" style={{ width: `${progress}%` }}>
              <div className="global-progress-glow"></div>
            </div>
          </div>
          <div className="loading-percentage">{Math.floor(progress)}%</div>
        </div>
      </div>
    </div>
  );
}
