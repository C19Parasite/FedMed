import { useState, useRef, useEffect, useCallback } from 'react';

interface Point {
  x: number;
  y: number;
}

/**
 * Checks whether point P lies inside the triangle formed by A, B, C.
 */
function isPointInTriangle(p: Point, a: Point, b: Point, c: Point): boolean {
  const v0 = [c.x - a.x, c.y - a.y];
  const v1 = [b.x - a.x, b.y - a.y];
  const v2 = [p.x - a.x, p.y - a.y];

  const dot00 = v0[0] * v0[0] + v0[1] * v0[1];
  const dot01 = v0[0] * v1[0] + v0[1] * v1[1];
  const dot02 = v0[0] * v2[0] + v0[1] * v2[1];
  const dot11 = v1[0] * v1[0] + v1[1] * v1[1];
  const dot12 = v1[0] * v2[0] + v1[1] * v2[1];

  const invDenom = 1 / (dot00 * dot11 - dot01 * dot01 || 0.0001);
  const u = (dot11 * dot02 - dot01 * dot12) * invDenom;
  const v = (dot00 * dot12 - dot01 * dot02) * invDenom;

  return u >= 0 && v >= 0 && u + v < 1;
}

export function useSafeTriangle(closeDelayMs = 280) {
  const [isOpen, setIsOpen] = useState(false);
  const triggerRef = useRef<HTMLElement | null>(null);
  const submenuRef = useRef<HTMLElement | null>(null);
  const lastMousePosRef = useRef<Point>({ x: 0, y: 0 });
  const closeTimerRef = useRef<number | null>(null);

  const clearCloseTimer = () => {
    if (closeTimerRef.current) {
      clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
  };

  const handleMouseMove = useCallback((e: MouseEvent) => {
    lastMousePosRef.current = { x: e.clientX, y: e.clientY };

    if (!isOpen || !submenuRef.current) return;

    const subRect = submenuRef.current.getBoundingClientRect();
    const currentPos = lastMousePosRef.current;

    // Triangle formed by cursor entry point and the top-right / bottom-right corners of submenu
    const cornerA: Point = { x: subRect.left, y: subRect.top };
    const cornerB: Point = { x: subRect.left, y: subRect.bottom };

    if (isPointInTriangle(currentPos, lastMousePosRef.current, cornerA, cornerB)) {
      clearCloseTimer();
    }
  }, [isOpen]);

  useEffect(() => {
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, [handleMouseMove]);

  const onTriggerMouseEnter = () => {
    clearCloseTimer();
    setIsOpen(true);
  };

  const onTriggerMouseLeave = () => {
    clearCloseTimer();
    closeTimerRef.current = window.setTimeout(() => {
      setIsOpen(false);
    }, closeDelayMs);
  };

  const onSubmenuMouseEnter = () => {
    clearCloseTimer();
    setIsOpen(true);
  };

  const onSubmenuMouseLeave = () => {
    clearCloseTimer();
    closeTimerRef.current = window.setTimeout(() => {
      setIsOpen(false);
    }, closeDelayMs);
  };

  return {
    isOpen,
    triggerRef,
    submenuRef,
    onTriggerMouseEnter,
    onTriggerMouseLeave,
    onSubmenuMouseEnter,
    onSubmenuMouseLeave,
  };
}
