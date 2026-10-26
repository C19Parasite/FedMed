import React from 'react';

export interface LogoProps {
  variant?: 'neural-brain' | 'brain-circuit' | 'medical-intelligence';
  size?: number | string;
  className?: string;
  primaryColor?: string;
  secondaryColor?: string;
}

/**
 * Neural Brain Vector Icon: Anatomical brain outline integrated with interconnected neural synaptic nodes
 */
export const NeuralBrainIcon: React.FC<Omit<LogoProps, 'variant'>> = ({
  size = 24,
  className = '',
}) => {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-label="FedMed Neural Brain AI"
    >
      <defs>
        <linearGradient id="neuralGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#FFFFFF" />
          <stop offset="100%" stopColor="rgba(255, 255, 255, 0.85)" />
        </linearGradient>
        <filter id="nodeGlow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="0.8" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
      </defs>

      {/* Left Hemisphere Outer Sulci Contour */}
      <path
        d="M15 6C13.2 6 11.5 6.8 10.3 8.2C8.8 8.4 7.2 9.6 6.6 11.2C5.9 13.1 6.3 15.2 7.4 16.7C6.8 18.2 7.1 20 8.2 21.3C9.3 22.7 11 23.5 12.8 23.8C13.5 25.1 14.8 26 16 26"
        stroke="url(#neuralGrad)"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Right Hemisphere Outer Sulci Contour */}
      <path
        d="M17 6C18.8 6 20.5 6.8 21.7 8.2C23.2 8.4 24.8 9.6 25.4 11.2C26.1 13.1 25.7 15.2 24.6 16.7C25.2 18.2 24.9 20 23.8 21.3C22.7 22.7 21 23.5 19.2 23.8C18.5 25.1 17.2 26 16 26"
        stroke="url(#neuralGrad)"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Inter-hemispheric Longitudinal Fissure */}
      <line
        x1="16"
        y1="7"
        x2="16"
        y2="24"
        stroke="url(#neuralGrad)"
        strokeWidth="1.2"
        strokeDasharray="2 2"
        opacity="0.6"
      />

      {/* Neural Synapse Interconnect Edges */}
      <line x1="11" y1="11" x2="16" y2="13" stroke="url(#neuralGrad)" strokeWidth="1" opacity="0.6" />
      <line x1="21" y1="11" x2="16" y2="13" stroke="url(#neuralGrad)" strokeWidth="1" opacity="0.6" />
      <line x1="16" y1="13" x2="11" y2="18" stroke="url(#neuralGrad)" strokeWidth="1" opacity="0.6" />
      <line x1="16" y1="13" x2="21" y2="18" stroke="url(#neuralGrad)" strokeWidth="1" opacity="0.6" />
      <line x1="11" y1="18" x2="16" y2="21" stroke="url(#neuralGrad)" strokeWidth="1" opacity="0.6" />
      <line x1="21" y1="18" x2="16" y2="21" stroke="url(#neuralGrad)" strokeWidth="1" opacity="0.6" />
      <line x1="9" y1="14" x2="11" y2="11" stroke="url(#neuralGrad)" strokeWidth="0.8" opacity="0.4" />
      <line x1="23" y1="14" x2="21" y2="11" stroke="url(#neuralGrad)" strokeWidth="0.8" opacity="0.4" />

      {/* Neural Synapse Nodes */}
      <circle cx="16" cy="13" r="2" fill="#FFFFFF" filter="url(#nodeGlow)" />
      <circle cx="11" cy="11" r="1.5" fill="#FFFFFF" />
      <circle cx="21" cy="11" r="1.5" fill="#FFFFFF" />
      <circle cx="9" cy="15" r="1.3" fill="url(#neuralGrad)" opacity="0.9" />
      <circle cx="23" cy="15" r="1.3" fill="url(#neuralGrad)" opacity="0.9" />
      <circle cx="11" cy="18" r="1.5" fill="#FFFFFF" />
      <circle cx="21" cy="18" r="1.5" fill="#FFFFFF" />
      <circle cx="16" cy="21" r="1.8" fill="#FFFFFF" filter="url(#nodeGlow)" />
    </svg>
  );
};

/**
 * Universal BrandLogo Component
 */
export const BrandLogo: React.FC<LogoProps> = ({
  size = 24,
  className = '',
}) => {
  return <NeuralBrainIcon size={size} className={className} />;
};
