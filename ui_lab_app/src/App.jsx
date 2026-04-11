import React, { useEffect, useRef, useState } from "react";
import { HeaderShell, JOURNEY_STEPS, ROUTES, SideNav, SupportRail } from "./components/common";
import LoadingScreen from "./screens/LoadingScreen";
import LandingScreen from "./screens/LandingScreen";
import JourneyScreen from "./screens/JourneyScreen";
import TradingWatchScreen from "./screens/TradingWatchScreen";
import ControlPlaneScreen from "./screens/ControlPlaneScreen";
import AboutScreen from "./screens/AboutScreen";
import RawDataScreen from "./screens/RawDataScreen";
import StrategiesScreen from "./screens/StrategiesScreen";
import { createInitialSystem } from "./system/mockSystem";
import { createSystemAdapter, mapStatusToProductShell } from "./system/systemAdapter";

function transportFromEnv() {
  const raw = String(import.meta.env.VITE_SYSTEM_TRANSPORT || "poll").toLowerCase().trim();
  return raw === "mock" || raw === "ws" ? raw : "poll";
}

export default function App() {
  const transport = transportFromEnv();
  const adapterRef = useRef(null);
  const [view, setView] = useState(ROUTES.loading);
  const [journeyIndex, setJourneyIndex] = useState(0);
  const [system, setSystem] = useState(createInitialSystem);
  const [selectedSymbol, setSelectedSymbol] = useState("BTCUSDm");
  const [transportState, setTransportState] = useState({
    mode: "booting",
    error: null,
  });
  const [busyAction, setBusyAction] = useState(null);
  const [lastActionResult, setLastActionResult] = useState(null);

  if (!adapterRef.current) {
    adapterRef.current = createSystemAdapter({
      transport,
      statusUrl: "/api/status",
      controlUrl: "/api/control",
      wsUrl: null,
      pollMs: 2000,
    });
  }

  useEffect(() => {
    const unsubscribe = adapterRef.current.subscribe(
      (payload) => {
        if (transport === "mock") {
          setSystem(payload);
        } else {
          setSystem((current) => mapStatusToProductShell(payload, current));
        }
        setTransportState({
          mode: "live",
          error: null,
        });
      },
      (error) => {
        setTransportState({
          mode: "degraded",
          error: error?.message || "transport error",
        });
      }
    );
    return () => {
      if (typeof unsubscribe === "function") unsubscribe();
    };
  }, [transport]);

  function nextJourneyStep() {
    setJourneyIndex((current) => {
      if (current >= JOURNEY_STEPS.length - 1) {
        setView(ROUTES.trading);
        return current;
      }
      return current + 1;
    });
  }

  function prevJourneyStep() {
    setJourneyIndex((current) => Math.max(0, current - 1));
  }

  useEffect(() => {
    const symbols = Array.from(
      new Set([
        ...(system.trading.lanes || []).map((lane) => lane.symbol),
        ...(system.registry.candidates || []).map((candidate) => candidate.symbol),
        system.registry.champion?.symbol,
        system.registry.canary?.symbol,
      ].filter(Boolean))
    );
    if (symbols.length && !symbols.includes(selectedSymbol)) {
      setSelectedSymbol(symbols[0]);
    }
  }, [system, selectedSymbol]);

  async function runAction(action, payload = {}) {
    try {
      setBusyAction(action);
      const result = await adapterRef.current.dispatch(action, payload);
      setLastActionResult({
        ...result,
        at: Date.now(),
      });
    } finally {
      setBusyAction(null);
    }
  }

  function startJourney() {
    setJourneyIndex(0);
    setView(ROUTES.journey);
  }

  function resetFlow() {
    setJourneyIndex(0);
    setView(ROUTES.landing);
    const blk = createInitialSystem();
    blk.trading.account.balance = 10000;
    blk.trading.account.equity = 10000;
    blk.trading.account.floatingPnl = 0;
    blk.trading.account.realizedToday = 0;
    blk.trading.account.openPositions = 0;
    blk.trading.lanes = [];
    blk.trading.tradeHistory = [];
    blk._history = { 
      equity: [10000], 
      pnl: [0],
      confidence: [0]
    };
    blk.incidents = [{ level: "info", title: "Day Zero Reset", message: "All session stats and historical memory purged. Starting from scratch." }];
    blk.timeline = [{ level: "info", text: "Manual system wipe executed. Session parity established." }];
    blk.__tick = 0;
    setSystem(blk);
    setJourneyIndex(0);
    setSelectedSymbol("BTCUSDm");
    runAction("reset_bot", {});
  }

  const journeyStep = JOURNEY_STEPS[Math.min(journeyIndex, JOURNEY_STEPS.length - 1)];
  const availableSymbols = Array.from(
    new Set([
      ...(system.trading.lanes || []).map((lane) => lane.symbol),
      ...(system.registry.candidates || []).map((candidate) => candidate.symbol),
      system.registry.champion?.symbol,
      system.registry.canary?.symbol,
    ].filter(Boolean))
  );

  if (view === ROUTES.loading) {
    return <LoadingScreen onComplete={() => setView(ROUTES.landing)} />;
  }

  return (
    <div className="app-shell">
      <div className="workspace-layout">
        <SideNav
          system={system}
          view={view}
          selectedSymbol={selectedSymbol}
          availableSymbols={availableSymbols}
          onChange={setView}
          onSelectSymbol={setSelectedSymbol}
          onStartJourney={startJourney}
          onGoTrading={() => setView(ROUTES.trading)}
          onGoControl={() => setView(ROUTES.control)}
          onReset={resetFlow}
        />

        <main className="main-stage">
          <HeaderShell
            system={system}
            view={view}
            journeyTitle={journeyStep.title}
            selectedSymbol={selectedSymbol}
            transport={transport}
            transportState={transportState}
            lastActionResult={lastActionResult}
          />

          {view === ROUTES.landing ? (
            <LandingScreen
              system={system}
              selectedSymbol={selectedSymbol}
              onStartJourney={startJourney}
              onGoTrading={() => setView(ROUTES.trading)}
              onGoControl={() => setView(ROUTES.control)}
            />
          ) : null}

          {view === ROUTES.journey ? (
            <JourneyScreen 
              system={system} 
              selectedSymbol={selectedSymbol} 
              activeIndex={journeyIndex} 
              onNext={nextJourneyStep} 
              onPrev={prevJourneyStep} 
            />
          ) : null}

          {view === ROUTES.trading ? (
            <TradingWatchScreen
              system={system}
              selectedSymbol={selectedSymbol}
              onReplayJourney={startJourney}
              onGoControl={() => setView(ROUTES.control)}
            />
          ) : null}

          {view === ROUTES.control ? (
            <ControlPlaneScreen
              system={system}
              busyAction={busyAction}
              onAction={runAction}
              onBackToTrading={() => setView(ROUTES.trading)}
            />
          ) : null}

          {view === ROUTES.about ? <AboutScreen /> : null}

          {view === ROUTES.raw ? <RawDataScreen system={system} /> : null}

          {view === ROUTES.strategies ? <StrategiesScreen system={system} /> : null}
        </main>

        <SupportRail system={system} selectedSymbol={selectedSymbol} />
      </div>
    </div>
  );
}
