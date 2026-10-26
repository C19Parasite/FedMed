/**
 * Helper to invoke the native View Transitions API with fallback and reduced-motion check.
 */
export function transitionView(updateCallback: () => void | Promise<void>) {
  // Check user preference for reduced motion
  const prefersReducedMotion =
    typeof window !== 'undefined' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Check if browser supports View Transitions API
  if (!prefersReducedMotion && typeof document !== 'undefined' && 'startViewTransition' in document) {
    // @ts-ignore - native API
    document.startViewTransition(() => {
      return updateCallback();
    });
  } else {
    updateCallback();
  }
}
