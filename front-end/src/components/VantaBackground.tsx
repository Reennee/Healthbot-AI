"use client";

import { useEffect } from "react";
import * as THREE from "three";
// @ts-expect-error - vanta has no types
import GLOBE from "vanta/dist/vanta.globe.min";

export default function VantaBackground() {
  useEffect(() => {
    const el = document.getElementById("vanta-bg");
    if (!el) return;
    // Initialize Vanta
    const effect = GLOBE({
      el,
      THREE,
      mouseControls: true,
      touchControls: true,
      gyroControls: false,
      minHeight: 200.0,
      minWidth: 200.0,
      scale: 1.0,
      scaleMobile: 1.0,
      color: 0x6366f1,
      backgroundColor: 0x111827,
      size: 0.8,
    });
    return () => {
      // Cleanup Vanta
      if (effect && typeof effect.destroy === "function") {
        effect.destroy();
      }
    };
  }, []);
  return null;
}


